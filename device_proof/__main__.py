import argparse
import json
from pathlib import Path
import sys

from pydantic import ValidationError

from .audit import MODEL_ID, AuditError, audit_model
from .backend import ProofBackendError
from .bundle import BundleError, export_bundle, verify_bundle
from .scoring import ROOT, ScoreRequest, score
from .statements import QuantizationError, build_statement


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "model-check":
        _model_check(args[1:])
        return
    if args and args[0] == "statement":
        _statement(args[1:])
        return
    if args and args[0] == "export-bundle":
        _export_bundle(args[1:])
        return
    if args and args[0] == "verify-bundle":
        _verify_bundle(args[1:])
        return
    parser = argparse.ArgumentParser(description="Run a device health score locally.")
    parser.add_argument("--input", type=Path, default=ROOT / "examples" / "sample.json")
    parsed = parser.parse_args(args)
    request = ScoreRequest.model_validate_json(parsed.input.read_text(encoding="utf-8"))
    try:
        result = score(request)
    except AuditError as exc:
        print(exc.code, file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


def _fail(code: str):
    # stderr carries only the stable error code: no paths, stack traces,
    # request contents, or artifact contents.
    print(code, file=sys.stderr)
    raise SystemExit(1)


def _statement(argv):
    parser = argparse.ArgumentParser(
        prog="device_proof statement",
        description="Emit the Q16 quantization statement for a score request.",
    )
    parser.add_argument("--input", type=Path, required=True)
    # argparse's default usage text would leak into stderr; keep only codes.
    parser.error = lambda message: _fail("invalid_arguments")
    parsed = parser.parse_args(argv)
    try:
        raw = parsed.input.read_text(encoding="utf-8")
    except OSError:
        _fail("input_unreadable")
    try:
        request = ScoreRequest.model_validate_json(raw)
        result = build_statement(request)
    except AuditError as exc:
        _fail(exc.code)
    except QuantizationError as exc:
        _fail(exc.code)
    except KeyError:
        _fail("unknown_model")
    except ValidationError:
        _fail("invalid_request")
    except Exception:
        _fail("internal_error")
    print(json.dumps(result.model_dump(), ensure_ascii=False))


def _export_bundle(argv):
    parser = argparse.ArgumentParser(
        prog="device_proof export-bundle",
        description="Export the versioned evidence bundle of a succeeded proof job.",
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "runtime")
    # argparse's default usage text would leak into stderr; keep only codes.
    parser.error = lambda message: _fail("invalid_arguments")
    parsed = parser.parse_args(argv)
    try:
        export_bundle(parsed.job_id, parsed.output, parsed.runtime_dir)
    except BundleError as exc:
        _fail(exc.code)
    except Exception:
        _fail("internal_error")


def _verify_bundle(argv):
    parser = argparse.ArgumentParser(
        prog="device_proof verify-bundle",
        description="Verify an evidence bundle standalone (no HTTP, no job store).",
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--backend-dir", type=Path, default=None)
    # argparse's default usage text would leak into stderr; keep only codes.
    parser.error = lambda message: _fail("invalid_arguments")
    parsed = parser.parse_args(argv)
    try:
        result = verify_bundle(parsed.bundle, backend_dir=parsed.backend_dir)
    except BundleError as exc:
        _fail(exc.code)
    except ProofBackendError as exc:
        _fail(exc.code)
    except Exception:
        _fail("internal_error")
    print(json.dumps(result, ensure_ascii=False))


def _model_check(argv):
    parser = argparse.ArgumentParser(
        prog="device_proof model-check",
        description="Audit the bundled model artifact and print a single-line JSON report.",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parsed = parser.parse_args(argv)
    try:
        report = audit_model(parsed.model_id)
    except AuditError as exc:
        # stderr carries only the stable error code: no paths, stack traces,
        # or artifact contents.
        print(exc.code, file=sys.stderr)
        raise SystemExit(1) from None
    except Exception:
        print("internal_error", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

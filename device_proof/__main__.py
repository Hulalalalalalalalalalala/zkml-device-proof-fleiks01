import argparse
import json
from pathlib import Path
import sys

from .audit import MODEL_ID, AuditError, audit_model
from .scoring import ROOT, ScoreRequest, score


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "model-check":
        _model_check(args[1:])
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

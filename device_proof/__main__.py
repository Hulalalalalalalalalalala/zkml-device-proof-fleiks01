import argparse
import json
import sys
from pathlib import Path

from .integrity import AUDIT_FAILED, MODEL_ID, IntegrityError
from .scoring import ROOT, ScoreRequest, guard, score


def _run_model_check(model_id: str) -> int:
    """Audit the fixed model and emit one single-line JSON report.

    On failure stderr carries exactly one stable error code (never a path,
    traceback or artifact content) and the process exits non-zero.
    """
    if model_id != MODEL_ID:
        print("unknown_model", file=sys.stderr)
        return 2
    try:
        report = guard.access().report
    except IntegrityError as error:
        print(error.code, file=sys.stderr)
        return 1
    except Exception:
        print(AUDIT_FAILED, file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0


def _run_score(input_path: Path) -> int:
    request = ScoreRequest.model_validate_json(input_path.read_text(encoding="utf-8"))
    print(json.dumps(score(request).model_dump(), indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the device health model.")
    subparsers = parser.add_subparsers(dest="command")

    check_parser = subparsers.add_parser(
        "model-check",
        help="Audit the fixed model's integrity and print a JSON report.",
    )
    check_parser.add_argument("--model-id", default=MODEL_ID)

    parser.add_argument(
        "--input", type=Path, default=ROOT / "examples" / "sample.json"
    )

    args = parser.parse_args(argv)
    if args.command == "model-check":
        return _run_model_check(args.model_id)
    return _run_score(args.input)


if __name__ == "__main__":
    raise SystemExit(main())

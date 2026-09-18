import argparse
import json
from pathlib import Path

from .scoring import ROOT, ScoreRequest, score


def main():
    parser = argparse.ArgumentParser(description="Run a device health score locally.")
    parser.add_argument("--input", type=Path, default=ROOT / "examples" / "sample.json")
    args = parser.parse_args()
    request = ScoreRequest.model_validate_json(args.input.read_text(encoding="utf-8"))
    print(json.dumps(score(request).model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

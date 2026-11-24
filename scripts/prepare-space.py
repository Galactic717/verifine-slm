import argparse
import json
import shutil
from pathlib import Path

from verifine.data.split import read_dataset
from verifine.eval.run_eval import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a static Hugging Face Space from paired evals"
    )
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("data/golden.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/space"))
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (args.before, args.after)]
    dataset, digest = read_dataset(args.dataset)
    if any(report["dataset_sha256"] != digest for report in reports):
        parser.error("Evaluation reports must match the supplied dataset SHA-256")
    if any(reports[0][key] != reports[1][key] for key in ("max_tokens", "constrained")):
        parser.error("Evaluation reports must use the same decoding mode and token limit")
    before, after = ({row["id"]: row for row in report["predictions"]} for report in reports)
    if before.keys() != after.keys():
        parser.error("Both reports must include the same case IDs")
    cases = [case for case in dataset if case.id in before]
    if not cases or not all(case.synthetic for case in cases):
        parser.error("The demo consumes only matched synthetic cases")
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    for filename in ("app.py", "README.md"):
        shutil.copyfile(root / "demo" / filename, args.output / filename)

    def select(row):
        return {key: row[key] for key in ("prediction", "schema_valid", "exact", "unsupported")}

    write_json(
        args.output / "examples.json",
        {
            "before_model": reports[0]["model"],
            "after_model": reports[1]["model"],
            "examples": [
                {
                    "id": case.id,
                    "language": case.language,
                    "task": case.task,
                    "document": case.document,
                    "expected": case.expected,
                    "before": select(before[case.id]),
                    "after": select(after[case.id]),
                }
                for case in cases
            ],
        },
    )


if __name__ == "__main__":
    main()

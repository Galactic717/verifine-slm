import argparse
import hashlib
import os
import re
import shutil
from pathlib import Path

from verifine.data.split import build_dataset
from verifine.eval.run_eval import write_json
from verifine.train.common import read_config


def check_base(path: Path, reference: str, revision: str | None, root: Path) -> None:
    match = re.fullmatch(r"/mnt/([a-zA-Z])/(.+)", reference)
    if os.name == "nt" and match:
        reference = f"{match[1]}:/{match[2]}"
    config = read_config(root / "configs/qwen3_4b_sft.yaml", None, None, None)
    pinned_id = reference == config.model and revision == config.revision
    recorded_path = Path(reference).resolve() == path.resolve()
    if not (pinned_id or recorded_path):
        raise ValueError(
            "Adapter must reference the supplied base snapshot or pinned model revision"
        )
    for record in (root / "configs" / "qwen3_4b_base.sha256").read_text().splitlines():
        expected, name = record.split()
        with (path / name).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Base shard does not match the pinned Qwen revision: {name}")


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description="Stage the Qwen adapter and public synthetic data")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True, help="Existing base model directory")
    parser.add_argument("--output", type=Path, default=Path("artifacts/release"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = read_config(root / "configs/qwen3_4b_sft.yaml", None, None, None)
    files = (
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
    license_file = args.base / "LICENSE"
    adapter = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if not license_file.is_file() or any(not (args.adapter / key).is_file() for key in files):
        parser.error("The base license and all Qwen adapter files are required")
    try:
        check_base(args.base, adapter["base_model_name_or_path"], adapter.get("revision"), root)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty release directory")
    target = args.output / "adapter"
    target.mkdir(parents=True, exist_ok=True)
    for filename in files:
        shutil.copyfile(args.adapter / filename, target / filename)
    adapter["base_model_name_or_path"] = base.model
    adapter["revision"] = base.revision
    write_json(target / "adapter_config.json", adapter)
    shutil.copyfile(license_file, target / "LICENSE")
    shutil.copyfile(root / "docs" / "MODEL_CARD.md", target / "README.md")
    shutil.copyfile(root / "docs" / "ABLATION_REPORT.md", target / "ABLATION_REPORT.md")
    build_dataset(args.output / "dataset", seed=42)
    shutil.copyfile(root / "docs" / "DATASET_CARD.md", args.output / "dataset" / "README.md")


if __name__ == "__main__":
    main()

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Choose an empty output directory: {path}")


def merge(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    empty(args.output)
    loader = AutoModelForImageTextToText if args.gemma4 else AutoModelForCausalLM
    model = loader.from_pretrained(args.base, local_files_only=True, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(args.adapter, local_files_only=True).save_pretrained(args.output)
    if args.gemma4:
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(args.base, local_files_only=True).save_pretrained(args.output)


def gguf(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as directory:
        temporary = Path(directory)
        intermediate = temporary / "unquantized.gguf"
        subprocess.run(
            [
                args.converter_python,
                str(args.converter),
                str(args.model),
                "--outfile",
                str(intermediate),
                "--outtype",
                "f16",
            ],
            check=True,
        )
        compressed = temporary / args.output.name
        subprocess.run(
            [str(args.quantizer), str(intermediate), str(compressed), "Q4_K_M"], check=True
        )
        shutil.move(compressed, args.output)


def compress(args: argparse.Namespace) -> None:
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform import AWQModifier
    from transformers import AutoTokenizer

    from verifine.data.prompts import messages
    from verifine.data.split import load_cases

    empty(args.output)
    kwargs = {}
    recipe = [
        QuantizationModifier(
            targets=["Linear"],
            ignore=["lm_head"],
            scheme="W4A16_ASYM" if args.command == "awq" else "FP8_DYNAMIC",
        )
    ]
    if args.command == "awq":
        if args.train.name != "train.jsonl":
            raise ValueError("Calibrate AWQ on train.jsonl")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        cases = load_cases(args.train)[: args.calibration_samples]
        if not cases:
            raise ValueError("AWQ requires non-empty training calibration data")
        texts = [
            tokenizer.apply_chat_template(
                messages(case.task, case.document),
                tokenize=False,
                add_generation_prompt=True,
            )
            for case in cases
        ]
        kwargs = {
            "dataset": Dataset.from_dict({"text": texts}),
            "num_calibration_samples": len(texts),
            "max_seq_length": 1024,
        }
        recipe.insert(0, AWQModifier())
    oneshot(model=str(args.model), recipe=recipe, output_dir=str(args.output), **kwargs)


def main() -> None:
    cli = argparse.ArgumentParser(description="Merge and quantize a local adapter")
    commands = cli.add_subparsers(dest="command", required=True)
    merge_cmd = commands.add_parser("merge")
    merge_cmd.add_argument("--base", type=Path, required=True)
    merge_cmd.add_argument("--adapter", type=Path, required=True)
    merge_cmd.add_argument("--output", type=Path, required=True)
    merge_cmd.add_argument("--gemma4", action="store_true")
    merge_cmd.set_defaults(run=merge)
    gguf_cmd = commands.add_parser("gguf")
    gguf_cmd.add_argument("--model", type=Path, required=True)
    gguf_cmd.add_argument("--converter", type=Path, required=True)
    gguf_cmd.add_argument("--converter-python", default=sys.executable)
    gguf_cmd.add_argument("--quantizer", type=Path, required=True)
    gguf_cmd.add_argument("--output", type=Path, required=True)
    gguf_cmd.set_defaults(run=gguf)
    for mode in ("awq", "fp8"):
        command = commands.add_parser(mode)
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if mode == "awq":
            command.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
            command.add_argument("--calibration-samples", type=int, default=128)
        command.set_defaults(run=compress)
    args = cli.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

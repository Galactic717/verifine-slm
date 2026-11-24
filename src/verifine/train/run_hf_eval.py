import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

from verifine.data.prompts import messages
from verifine.data.schemas import Case
from verifine.data.split import read_dataset
from verifine.eval.run_eval import describe, make_row, write_json
from verifine.train.common import Settings, load_qlora, read_config


def evaluate(
    args: argparse.Namespace,
    settings: Settings,
    cases: list[Case],
    digest: str,
    model: Any,
    tokenizer: Any,
    label: str,
    output: Path,
) -> None:
    import torch

    model.eval()
    # Use greedy decoding for all members of the ablation under identical prompts and token limits.
    model.generation_config.do_sample = False
    for name in ("top_p", "top_k", "temperature", "min_p"):
        setattr(model.generation_config, name, None)
    tokenizer.padding_side = "left"
    rows = []
    eos = model.generation_config.eos_token_id
    end_ids = set(eos if isinstance(eos, list) else [eos, tokenizer.eos_token_id])
    with torch.inference_mode():
        for start in range(0, len(cases), args.batch_size):
            batch = cases[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    messages(case.task, case.document),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for case in batch
            ]
            inputs = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
            inputs = inputs.to(model.device)
            torch.cuda.synchronize()
            clock = perf_counter()
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )[:, inputs["input_ids"].shape[-1] :]
            torch.cuda.synchronize()
            milliseconds = (perf_counter() - clock) * 1000
            counts = inputs["attention_mask"].sum(dim=1).tolist()
            for case, ids, count in zip(batch, generated.tolist(), counts, strict=True):
                end = next((i + 1 for i, token in enumerate(ids) if token in end_ids), len(ids))
                text = tokenizer.decode(ids[:end], skip_special_tokens=True)
                rows.append(
                    make_row(
                        case,
                        text,
                        milliseconds if args.batch_size == 1 else None,
                        None,
                        batch_elapsed_ms=milliseconds,
                        prompt_tokens=count,
                        completion_tokens=end,
                        finish_reason="stop" if ids[end - 1] in end_ids else "length",
                    )
                )
            print(f"{label}: {len(rows)}/{len(cases)}", flush=True)
    write_json(
        output,
        {
            "model": label,
            "backend": "transformers-nf4",
            "batch_size": args.batch_size,
            "constrained": False,
            "max_tokens": args.max_tokens,
            "seed": settings.seed,
            "dataset_sha256": digest,
            "summary": describe(rows),
            "predictions": rows,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare adapters with identical NF4 inference")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--ablation", nargs=2, type=Path, metavar=("SFT", "GRPO"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/golden.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=240)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_tokens < 1 or args.limit < 0:
        parser.error("Invalid batch size, output length or limit")
    if args.ablation and args.adapter:
        parser.error("Use either --adapter or --ablation")
    if args.ablation and any(
        not (path / "adapter_config.json").is_file() for path in args.ablation
    ):
        parser.error("Ablation requires both saved adapters")
    settings = read_config(args.config, args.model, None, None)
    cases, digest = read_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        parser.error("No cases to evaluate")
    model, tokenizer = load_qlora(settings, existing=args.adapter, inference=True)
    if not args.ablation:
        evaluate(args, settings, cases, digest, model, tokenizer, args.label, args.output)
        return

    from peft import PeftModel

    evaluate(
        args,
        settings,
        cases,
        digest,
        model,
        tokenizer,
        f"{args.label} base",
        args.output / "base.json",
    )
    sft, grpo = args.ablation
    model = PeftModel.from_pretrained(model, str(sft), adapter_name="sft", local_files_only=True)
    evaluate(
        args,
        settings,
        cases,
        digest,
        model,
        tokenizer,
        f"{args.label} SFT",
        args.output / "sft.json",
    )
    model.load_adapter(str(grpo), adapter_name="grpo", local_files_only=True)
    model.set_adapter("grpo")
    evaluate(
        args,
        settings,
        cases,
        digest,
        model,
        tokenizer,
        f"{args.label} SFT + GRPO",
        args.output / "grpo.json",
    )


if __name__ == "__main__":
    main()

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import cast

from verifine.serve.client import KINDS, Config, Kind


def backend_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=KINDS, default="llama")
    parser.add_argument("--url", default=Config.url)
    parser.add_argument("--model", default=Config.model)
    parser.add_argument("--api-key-env", default="VERIFINE_API_KEY")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=240)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--dataset", type=Path, default=Path("data/golden.jsonl"))
    parser.add_argument("--output", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifine SLM")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="Build and audit bilingual splits")
    data.add_argument("--output", type=Path, default=Path("data"))
    data.add_argument("--source", type=Path)
    data.add_argument("--seed", type=int, default=42)
    quality = commands.add_parser("eval", help="Evaluate quality against a JSONL split")
    backend_options(quality)
    quality.add_argument("--limit", type=int, default=0)
    quality.add_argument("--allow-remote", action="store_true")
    quality.add_argument("--logprobs", action="store_true", help="Request token confidence")
    quality.add_argument("--input-usd-per-mtok", type=float)
    quality.add_argument("--output-usd-per-mtok", type=float)
    serving = commands.add_parser("bench", help="Benchmark streaming throughput and GPU use")
    backend_options(serving)
    serving.add_argument("--count", type=int, default=48)
    serving.add_argument("--warmup", type=int, default=8)
    serving.add_argument("--gpu", type=int, default=0)
    serving.add_argument("--usd-kwh", type=float)
    comparison = commands.add_parser("compare", help="Compute paired bootstrap CIs and McNemar")
    comparison.add_argument("before", type=Path)
    comparison.add_argument("after", type=Path)
    comparison.add_argument("--draws", type=int, default=1000)
    comparison.add_argument("--output", type=Path, required=True)
    gateway = commands.add_parser("gateway", help="Serve structured extraction and chat proxy")
    gateway.add_argument("--host", default="127.0.0.1")
    gateway.add_argument("--port", type=int, default=8088)
    arguments = parser.parse_args()
    try:
        run(arguments)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


def run(arguments: argparse.Namespace) -> None:
    if arguments.command == "data":
        from verifine.data.split import build_dataset

        result = build_dataset(arguments.output, arguments.seed, arguments.source)
    elif arguments.command == "compare":
        from verifine.eval.run_eval import compare

        result = compare(arguments.before, arguments.after, arguments.output, arguments.draws)
    elif arguments.command == "gateway":
        import uvicorn

        from verifine.serve.gateway import create_app

        uvicorn.run(create_app(), host=arguments.host, port=arguments.port, access_log=False)
        return
    else:
        from verifine.eval.bench_serving import benchmark
        from verifine.eval.run_eval import evaluate

        cfg = Config(
            model=arguments.model,
            url=arguments.url,
            kind=cast(Kind, arguments.backend),
            api_key=os.getenv(arguments.api_key_env, ""),
            timeout=arguments.timeout,
            input_usd=getattr(arguments, "input_usd_per_mtok", None),
            output_usd=getattr(arguments, "output_usd_per_mtok", None),
        )
        if arguments.command == "eval":
            result = asyncio.run(
                evaluate(
                    cfg,
                    arguments.dataset,
                    arguments.output,
                    limit=arguments.limit,
                    concurrency=arguments.concurrency,
                    constrained=arguments.constrained,
                    allow_remote=arguments.allow_remote,
                    logprobs=arguments.logprobs,
                    seed=arguments.seed,
                    max_tokens=arguments.max_tokens,
                )
            )["summary"]
        else:
            result = asyncio.run(
                benchmark(
                    cfg,
                    arguments.dataset,
                    arguments.output,
                    count=arguments.count,
                    concurrency=arguments.concurrency,
                    warmup=arguments.warmup,
                    constrained=arguments.constrained,
                    gpu=arguments.gpu,
                    usd_kwh=arguments.usd_kwh,
                    max_tokens=arguments.max_tokens,
                    seed=arguments.seed,
                )
            )["summary"]
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if result.get("errors", 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

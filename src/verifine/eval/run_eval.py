import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from verifine.data.prompts import messages
from verifine.data.schemas import SCHEMAS, Case
from verifine.data.split import read_dataset
from verifine.eval.stats import Row, paired, summarize
from verifine.rewards.composite import score
from verifine.serve.client import Backend, Config

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def check_data_boundary(cases: list[Case], config: Config, allow_remote: bool) -> None:
    if config.host in LOOPBACK and config.kind not in {"openai", "anthropic"}:
        return
    if not allow_remote or any(not case.synthetic for case in cases):
        raise ValueError("Remote evaluation requires --allow-remote and synthetic cases only")
    if config.kind in {"openai", "anthropic"} and not config.api_key:
        raise ValueError("Set the backend API key in the environment")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def make_row(
    case: Case,
    prediction: str,
    elapsed_ms: float | None,
    error: str | None,
    **metadata: Any,
) -> Row:
    return {
        "id": case.id,
        "language": case.language,
        "task": case.task,
        "expected": case.expected,
        "prediction": prediction,
        "elapsed_ms": elapsed_ms,
        "error": error,
        **metadata,
        **asdict(score(case, prediction)),
    }


def describe(rows: list[Row]) -> Row:
    prices = [row.get("estimated_cost_usd") for row in rows]
    known_prices = [float(price) for price in prices if price is not None]
    total_price = sum(known_prices) if prices and len(known_prices) == len(prices) else None
    return {
        **summarize(rows),
        "by_language": {
            key: summarize([r for r in rows if r["language"] == key]) for key in ("uk", "en")
        },
        "by_task": {key: summarize([r for r in rows if r["task"] == key]) for key in SCHEMAS},
        "estimated_cost_usd": total_price,
        "estimated_usd_per_million_requests": total_price * 1e6 / len(rows)
        if total_price is not None
        else None,
    }


async def evaluate(
    config: Config,
    source: Path,
    output: Path,
    *,
    limit: int,
    concurrency: int,
    constrained: bool,
    allow_remote: bool,
    seed: int,
    max_tokens: int,
    logprobs: bool = False,
) -> Row:
    if concurrency < 1 or limit < 0 or max_tokens < 1:
        raise ValueError("Concurrency and max tokens must be positive. Limit must be non-negative")
    cases, digest = read_dataset(source)
    if limit:
        cases = cases[:limit]
    if not cases:
        raise ValueError("Evaluation requires at least one case")
    check_data_boundary(cases, config, allow_remote)
    if logprobs and config.kind in {"ollama", "anthropic"}:
        raise ValueError("This backend does not provide OpenAI-compatible token logprobs")
    slots = asyncio.Semaphore(concurrency)
    async with Backend(config) as backend:

        async def run(case: Case) -> Row:
            async with slots:
                try:
                    answer = await backend.complete(
                        messages(case.task, case.document),
                        schema=SCHEMAS[case.task].model_json_schema() if constrained else None,
                        max_tokens=max_tokens,
                        seed=seed,
                        logprobs=logprobs,
                    )
                except httpx.HTTPStatusError as error:
                    return make_row(case, "", None, f"HTTP {error.response.status_code}")
                except (httpx.RequestError, TypeError, ValueError, KeyError, IndexError) as error:
                    return make_row(case, "", None, type(error).__name__)
            return make_row(
                case,
                answer.text,
                answer.elapsed_ms,
                None,
                prompt_tokens=answer.prompt_tokens,
                completion_tokens=answer.completion_tokens,
                finish_reason=answer.finish_reason,
                confidence=answer.confidence,
                estimated_cost_usd=config.price(answer.prompt_tokens, answer.completion_tokens),
            )

        rows = await asyncio.gather(*(run(case) for case in cases))
    payload = {
        "model": config.model,
        "backend": config.kind,
        "constrained": constrained,
        "seed": seed,
        "max_tokens": max_tokens,
        "confidence_method": "geometric_mean_token_probability" if logprobs else None,
        "concurrency": concurrency,
        "dataset_sha256": digest,
        "summary": describe(rows),
        "predictions": rows,
    }
    write_json(output, payload)
    return payload


def compare(before: Path, after: Path, output: Path, draws: int) -> Row:
    left, right = [json.loads(path.read_text(encoding="utf-8")) for path in (before, after)]
    for key in ("dataset_sha256", "constrained", "max_tokens"):
        if left[key] != right[key]:
            raise ValueError(f"Paired evaluation requires matching {key}")
    result = paired(left["predictions"], right["predictions"], draws)
    write_json(output, {"before": left["model"], "after": right["model"], **result})
    return result

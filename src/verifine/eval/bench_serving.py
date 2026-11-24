import asyncio
import csv
import shutil
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import httpx

from verifine.data.prompts import messages
from verifine.data.schemas import SCHEMAS
from verifine.data.split import read_dataset
from verifine.eval.run_eval import check_data_boundary, write_json
from verifine.eval.stats import quantile
from verifine.rewards.schema_reward import parse_output
from verifine.serve.client import Backend, Config


async def sample_gpu(stop: asyncio.Event, gpu: int, start: float) -> list[dict[str, float | None]]:
    executable = shutil.which("nvidia-smi")
    samples: list[dict[str, float | None]] = []
    if not executable:
        return samples
    while not stop.is_set():
        process = await asyncio.create_subprocess_exec(
            executable,
            f"--id={gpu}",
            "--query-gpu=memory.used,power.draw",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        output, _ = await process.communicate()
        if process.returncode != 0 or not output.strip():
            return samples
        cells = next(csv.reader([output.decode().splitlines()[0]], skipinitialspace=True))
        try:
            vram, power = [float(item) if "[" not in item else None for item in cells]
        except ValueError:
            return samples
        samples.append({"elapsed_s": perf_counter() - start, "vram_mib": vram, "watts": power})
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.25)
        except TimeoutError:
            continue
    return samples


async def benchmark(
    config: Config,
    source: Path,
    output: Path,
    *,
    count: int,
    concurrency: int,
    warmup: int,
    constrained: bool,
    gpu: int,
    usd_kwh: float | None,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    if min(count, concurrency, max_tokens) < 1 or min(warmup, gpu) < 0:
        raise ValueError("Invalid benchmark counts, GPU index, or generation length")
    if usd_kwh is not None and (not isfinite(usd_kwh) or usd_kwh < 0):
        raise ValueError("Electricity price must be finite and non-negative")
    cases, digest = read_dataset(source)
    if not cases or not all(case.synthetic for case in cases):
        raise ValueError("Serving benchmarks require non-empty synthetic cases")
    check_data_boundary(cases, config, allow_remote=False)
    slots = asyncio.Semaphore(concurrency)
    async with Backend(config) as backend:

        async def run(index: int) -> dict[str, Any]:
            case = cases[index % len(cases)]
            async with slots:
                try:
                    row = await backend.complete(
                        messages(case.task, case.document),
                        stream=True,
                        max_tokens=max_tokens,
                        seed=seed,
                        schema=SCHEMAS[case.task].model_json_schema() if constrained else None,
                    )
                    return {
                        "id": case.id,
                        "error": None,
                        "schema_valid": parse_output(case.task, row.text)[1] is not None,
                        **{key: value for key, value in asdict(row).items() if key != "text"},
                    }
                except (httpx.HTTPError, KeyError, ValueError, TypeError, IndexError) as error:
                    return {"id": case.id, "error": type(error).__name__}

        for idx in range(warmup):
            result = await run(idx)
            if result["error"]:
                raise RuntimeError(f"Serving warmup failed: {result['error']}")
        stopped = asyncio.Event()
        started = perf_counter()
        sampler = asyncio.create_task(sample_gpu(stopped, gpu, started))
        try:
            rows = await asyncio.gather(*(run(i + warmup) for i in range(count)))
        finally:
            elapsed = perf_counter() - started
            stopped.set()
            samples = await sampler
    successful = [row for row in rows if not row["error"]]
    vram = [row["vram_mib"] for row in samples if row["vram_mib"] is not None]
    power = [row["watts"] for row in samples if row["watts"] is not None]
    total_output = sum(row["completion_tokens"] or 0 for row in successful)
    total_input = sum(row["prompt_tokens"] or 0 for row in successful)
    token_counts_complete = bool(successful) and all(
        row["completion_tokens"] is not None and row["prompt_tokens"] is not None
        for row in successful
    )
    energy_cost = mean(power) * elapsed * usd_kwh / 3.6e6 if power and usd_kwh is not None else None
    summary: dict[str, Any] = {
        "successful": len(successful),
        "errors": count - len(successful),
        "wall_s": elapsed,
        "requests_per_s": len(successful) / elapsed,
        "schema_valid_requests_per_s": sum(row["schema_valid"] for row in successful) / elapsed,
        "length_limited": sum(row["finish_reason"] == "length" for row in successful),
        "output_tokens_per_s": total_output / elapsed if token_counts_complete else None,
        "peak_board_vram_mib": max(vram) if vram else None,
        "board_power_sample_mean_w": mean(power) if power else None,
        "electricity_usd_per_kwh_assumption": usd_kwh,
        "gpu_electricity_usd_per_million_total_tokens": energy_cost
        * 1e6
        / (total_output + total_input)
        if energy_cost is not None and token_counts_complete and total_input + total_output
        else None,
        "gpu_electricity_usd_per_million_requests": energy_cost * 1e6 / len(successful)
        if energy_cost is not None and successful
        else None,
    }
    for name in ("ttft_ms", "answer_ttft_ms", "tpot_ms", "elapsed_ms"):
        values = [row[name] for row in successful if row[name] is not None]
        for label, quant in (("p50", 0.5), ("p95", 0.95)):
            summary[f"{name}_{label}"] = quantile(values, quant)
    report = {
        "model": config.model,
        "backend": config.kind,
        "count": count,
        "warmup": warmup,
        "concurrency": concurrency,
        "constrained": constrained,
        "max_tokens": max_tokens,
        "seed": seed,
        "dataset_sha256": digest,
        "summary": summary,
        "requests": rows,
        "gpu_samples": samples,
    }
    write_json(output, report)
    return report

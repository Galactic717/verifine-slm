from collections import Counter, defaultdict
from math import comb, isfinite
from random import Random
from statistics import mean
from typing import Any

Row = dict[str, Any]


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    index = int(position)
    return ordered[index] + (ordered[min(index + 1, len(ordered) - 1)] - ordered[index]) * (
        position - index
    )


def field_f1(rows: list[Row], key: str = "fields") -> float | None:
    buckets: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        for field, outcome in row[key].items():
            buckets[row["task"], field][outcome] += 1
    scores = []
    for bucket in buckets.values():
        denominator = 2 * bucket["tp"] + bucket["fp"] + bucket["fn"] + 2 * bucket["wrong"]
        if denominator:
            scores.append(2 * bucket["tp"] / denominator)
    return mean(scores) if scores else None


def summarize(rows: list[Row]) -> Row:
    asserted = sum(row["asserted"] for row in rows)
    absent = sum(row.get("absent_fields", 0) for row in rows)
    calibration = [
        row
        for row in rows
        if row.get("confidence") is not None
        and isfinite(row["confidence"])
        and 0 <= row["confidence"] <= 1
    ]
    brier = (
        mean((row["confidence"] - row["exact"]) ** 2 for row in calibration)
        if calibration
        else None
    )
    return {
        "n": len(rows),
        "errors": sum(row.get("error") is not None for row in rows),
        "json_valid": mean(row["json_valid"] for row in rows) if rows else None,
        "schema_valid": mean(row["schema_valid"] for row in rows) if rows else None,
        "field_macro_f1": field_f1(rows),
        "fuzzy_field_macro_f1": field_f1(rows, "fuzzy_fields")
        if rows and "fuzzy_fields" in rows[0]
        else None,
        "all_fields_exact": mean(row["exact"] for row in rows) if rows else None,
        "unsupported_value_rate": sum(row["unsupported"] for row in rows) / asserted
        if asserted
        else None,
        "abstention_recall": sum(row.get("null_correct", 0) for row in rows) / absent
        if absent
        else None,
        "p95_latency_ms": quantile(
            [row["elapsed_ms"] for row in rows if row.get("elapsed_ms") is not None],
            0.95,
        ),
        "brier_exact": brier,
        "confidence_coverage": len(calibration) / len(rows) if rows else 0,
    }


def paired(first: list[Row], second: list[Row], draws: int = 1000, seed: int = 42) -> Row:
    left, right = ({row["id"]: row for row in group} for group in (first, second))
    if (
        not left
        or len(left) != len(first)
        or len(right) != len(second)
        or left.keys() != right.keys()
        or draws < 1
    ):
        raise ValueError("Paired comparison requires the same non-empty set of unique case IDs")
    ids = sorted(left)
    deltas = [int(right[key]["exact"]) - int(left[key]["exact"]) for key in ids]
    gains, losses = deltas.count(1), deltas.count(-1)
    discordant = gains + losses
    p_value = min(
        1.0, 2 * sum(comb(discordant, k) / (2**discordant) for k in range(min(gains, losses) + 1))
    )
    rng = Random(seed)
    bootstrap = []
    field_bootstrap = []
    for _ in range(draws):
        sampled = rng.choices(ids, k=len(ids))
        bootstrap.append(mean(int(right[k]["exact"]) - int(left[k]["exact"]) for k in sampled))
        former, latter = (field_f1([group[k] for k in sampled]) for group in (left, right))
        if former is not None and latter is not None:
            field_bootstrap.append(latter - former)

    def ci(values: list[float]) -> list[float | None]:
        return [quantile(values, 0.025), quantile(values, 0.975)]

    former, latter = field_f1(first), field_f1(second)
    return {
        "cases": len(ids),
        "exact_delta": mean(deltas),
        "exact_bootstrap_95": ci(bootstrap),
        "field_macro_f1_delta": latter - former
        if former is not None and latter is not None
        else None,
        "field_macro_f1_bootstrap_95": ci(field_bootstrap),
        "mcnemar_exact_p": p_value,
        "exact_gains": gains,
        "exact_losses": losses,
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }

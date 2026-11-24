from dataclasses import dataclass, field
from difflib import SequenceMatcher

from verifine.data.schemas import Case
from verifine.rewards.grounding import equal_value, is_supported, leaves, normalize
from verifine.rewards.schema_reward import parse_output


@dataclass(frozen=True)
class Score:
    json_valid: bool
    schema_valid: bool
    total: float = 0
    grounding: float = 0
    abstention: float | None = None
    accuracy: float = 0
    tool_args: float | None = None
    asserted: int = 0
    unsupported: int = 0
    exact: bool = False
    fields: dict[str, str] = field(default_factory=dict)
    fuzzy_fields: dict[str, str] = field(default_factory=dict)
    absent_fields: int = 0
    null_correct: int = 0


def score(case: Case, completion: str) -> Score:
    valid_json, parsed = parse_output(case.task, completion)
    expected = leaves(case.expected)
    absent = sum(value is None for value in expected.values())
    if parsed is None:
        missing = {key: "fn" for key, value in expected.items() if value is not None}
        return Score(valid_json, False, fields=missing, fuzzy_fields=missing, absent_fields=absent)
    predicted = leaves(parsed.model_dump())
    fields = {}
    fuzzy_fields = {}
    correct = asserted = unsupported = null_correct = 0
    for key, gold in expected.items():
        actual = predicted[key]
        match = equal_value(actual, gold)
        correct += match
        if actual is not None:
            asserted += 1
            unsupported += not is_supported(case.document, key, actual, predicted.get("currency"))
        if gold is None and actual is None:
            null_correct += 1
            continue
        outcome = "tp" if match else "fp" if gold is None else "fn" if actual is None else "wrong"
        fields[key] = outcome
        approximate = (
            key in {"supplier", "counterparty", "jurisdiction", "symptom", "documented_finding"}
            and isinstance(actual, str)
            and isinstance(gold, str)
            and SequenceMatcher(None, normalize(actual), normalize(gold)).ratio() >= 0.9
        )
        fuzzy_fields[key] = "tp" if approximate else outcome
    accuracy = correct / len(expected)
    grounding = 1 - unsupported / asserted if asserted else float(absent == len(expected))
    abstention = null_correct / absent if absent else None
    tool_args = accuracy if case.task == "support" else None
    parts = [(0.15, 1.0), (0.50, accuracy), (0.20, grounding)]
    if abstention is not None:
        parts.append((0.15, abstention))
    if tool_args is not None:
        parts.append((0.20, tool_args))
    total = sum(weight * part for weight, part in parts) / sum(weight for weight, _ in parts)
    return Score(
        valid_json,
        True,
        total,
        grounding,
        abstention,
        accuracy,
        tool_args,
        asserted,
        unsupported,
        correct == len(expected),
        fields,
        fuzzy_fields,
        absent,
        null_correct,
    )

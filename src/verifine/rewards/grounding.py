import re
import unicodedata
from collections.abc import Iterable
from decimal import Decimal

from verifine.data.schemas import Record, Value

TOOL_CUES: dict[str, tuple[str, ...]] = {
    "assign_priority": (
        "change priority",
        "set ticket priority",
        "raise the priority",
        "priority update",
        "змінити пріоритет",
        "встановити пріоритет",
        "підняти пріоритет",
        "зміна пріоритету",
    ),
    "lookup_ticket": (
        "ticket status",
        "check the status",
        "status lookup",
        "status for ticket",
        "статус заявки",
        "перевірити стан",
        "перевірка статусу",
        "статус тікета",
    ),
}

PRIORITY_CUES: dict[str, tuple[str, ...]] = {
    "high": ("high", "високий"),
    "normal": ("normal", "звичайний"),
    "low": ("low", "низький"),
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'").replace("ʼ", "'")
    return " ".join(text.split())


def leaves(record: Record) -> dict[str, Value]:
    fields: dict[str, Value] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            fields.update({f"{key}.{sub}": item for sub, item in value.items()})
        else:
            fields[key] = value
    return fields


def contains(text: str, values: Iterable[str]) -> bool:
    return any(re.search(r"(?<!\w)" + re.escape(normalize(v)) + r"(?!\w)", text) for v in values)


def is_supported(document: str, field: str, value: str | int, currency: Value = None) -> bool:
    text = normalize(document)
    if field == "total_minor":
        amounts = re.findall(r"(?<![\w.,-])(\d+(?:[.,]\d{2})?)\s+(uah|usd|eur)\b", text)
        return any(
            int(Decimal(item.replace(",", ".")) * 100) == value
            and (currency is None or str(currency).casefold() == code)
            for item, code in amounts
        )
    if field.endswith("_days"):
        numbers = re.findall(r"(?<![\w.,-])(\d+)\s+(?:days?|день|днів|дні)\b", text)
        return any(int(item) == value for item in numbers)
    if field == "tool":
        return contains(text, TOOL_CUES.get(str(value), ()))
    if field == "arguments.priority":
        return contains(text, PRIORITY_CUES.get(str(value), ()))
    return contains(text, (str(value),))


def equal_value(actual: Value, expected: Value) -> bool:
    if isinstance(actual, str) and isinstance(expected, str):
        return normalize(actual) == normalize(expected)
    return actual == expected

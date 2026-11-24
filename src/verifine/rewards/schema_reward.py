import json
from typing import Any

from pydantic import ValidationError

from verifine.data.schemas import SCHEMAS, StrictModel, Task


def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def reject_constant(token: str) -> None:
    raise ValueError(f"Non-JSON constant: {token}")


def parse_output(task: Task, text: str) -> tuple[bool, StrictModel | None]:
    try:
        payload = json.loads(text, object_pairs_hook=unique_keys, parse_constant=reject_constant)
    except (ValueError, RecursionError):
        return False, None
    try:
        return True, SCHEMAS[task].model_validate(payload)
    except ValidationError:
        return True, None

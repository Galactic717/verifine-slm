import hashlib
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from datasketch import MinHash, MinHashLSH

from verifine.data.schemas import Case, Task
from verifine.data.synthesize import synthetic_cases
from verifine.rewards.composite import score
from verifine.rewards.grounding import normalize

THRESHOLD = 0.84
SPLITS = ("train", "validation", "golden")
PII = re.compile(
    r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|"
    r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-])"
    r"\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\w)|\b\d{10,16}\b"
)


def clean(text: str) -> str:
    return PII.sub("[REDACTED]", text.replace("\r\n", "\n")).strip()


def shingles(text: str) -> set[str]:
    tokens = re.findall(r"[\w'-]+", normalize(text))
    return {" ".join(tokens[i : i + 3]) for i in range(max(1, len(tokens) - 2))}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right)


def audit(splits: dict[str, list[Case]]) -> None:
    groups: dict[str, str] = {}
    ids: set[str] = set()
    encoded: dict[str, list[tuple[Case, set[str]]]] = {}
    for name, cases in splits.items():
        encoded[name] = []
        for case in cases:
            if case.id in ids:
                raise ValueError(f"Duplicate case ID: {case.id}")
            ids.add(case.id)
            if groups.setdefault(case.group, name) != name:
                raise ValueError(f"Cross-split group: {case.group}")
            encoded[name].append((case, shingles(case.document)))
    # Exhaustive cross-split validation catches pairs missed by approximate MinHash lookup.
    for first, second in combinations(encoded.values(), 2):
        for a, a_shingles in first:
            for b, b_shingles in second:
                if jaccard(a_shingles, b_shingles) >= THRESHOLD:
                    raise ValueError(f"Cross-split near duplicate: {a.id}, {b.id}")


def load_cases(path: Path) -> list[Case]:
    return parse_cases(path.read_bytes())


def parse_cases(payload: bytes) -> list[Case]:
    return [
        Case.model_validate_json(line)
        for line in payload.decode("utf-8-sig").split("\n")
        if line.strip()
    ]


def read_dataset(path: Path) -> tuple[list[Case], str]:
    payload = path.read_bytes()
    return parse_cases(payload), hashlib.sha256(payload).hexdigest()


def deduplicate(cases: list[Case]) -> list[Case]:
    index = MinHashLSH(threshold=0.75, num_perm=64)
    accepted: dict[str, tuple[Case, set[str]]] = {}
    seen_ids: set[str] = set()
    seen_documents: dict[tuple[Task, str], Case] = {}
    for case in cases:
        if case.id in seen_ids:
            raise ValueError(f"Duplicate case ID: {case.id}")
        seen_ids.add(case.id)
        normalized = normalize(case.document)
        document_key = case.task, normalized
        duplicate = seen_documents.get(document_key)
        if duplicate is not None:
            if duplicate.expected != case.expected:
                raise ValueError(f"Conflicting labels for the same task: {duplicate.id}, {case.id}")
            continue
        seen_documents[document_key] = case
        tokens = shingles(normalized)
        key = MinHash(num_perm=64)
        key.update_batch([word.encode() for word in sorted(tokens)])
        if any(
            accepted[match][0].task == case.task
            and jaccard(tokens, accepted[match][1]) >= THRESHOLD
            for match in index.query(key)
        ):
            continue
        index.insert(case.id, key)
        accepted[case.id] = (case, tokens)
    return [case for case, _ in accepted.values()]


def build_dataset(directory: Path, seed: int, source: Path | None = None) -> dict[str, Any]:
    splits: dict[str, list[Case]] = defaultdict(list)
    for case, name in synthetic_cases(seed):
        splits[name].append(case)
    if source:
        for original in load_cases(source):
            case = Case.model_validate(
                {
                    **original.model_dump(),
                    "document": clean(original.document),
                    "synthetic": False,
                }
            )
            if score(case, json.dumps(case.expected)).total < 1:
                raise ValueError(
                    f"Gold labels fail schema/evidence checks after scrubbing: {case.id}"
                )
            splits["train"].append(case)
    for name in SPLITS:
        splits[name] = deduplicate(splits[name])
        for case in splits[name]:
            if score(case, json.dumps(case.expected)).total < 1:
                raise ValueError(f"Invalid or unsupported gold label: {case.id}")
    if len(splits["golden"]) != 300:
        raise ValueError("The golden set must contain 300 unique documents")
    audit(splits)
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for name in SPLITS:
        target = directory / f"{name}.jsonl"
        payload = "".join(case.model_dump_json() + "\n" for case in splits[name]).encode()
        target.write_bytes(payload)
        files[target.name] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "seed": seed,
        "counts": {name: len(splits[name]) for name in SPLITS},
        "sha256": files,
        "jaccard_threshold": THRESHOLD,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest

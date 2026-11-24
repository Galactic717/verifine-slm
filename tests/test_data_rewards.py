import json
from pathlib import Path

import pytest

from verifine.data.schemas import Case
from verifine.data.split import audit, build_dataset, deduplicate, load_cases
from verifine.data.synthesize import synthetic_cases
from verifine.rewards.composite import score
from verifine.rewards.schema_reward import parse_output


@pytest.fixture
def invoice() -> Case:
    return Case(
        id="invoice-test",
        group="manual",
        task="invoice",
        language="uk",
        synthetic=True,
        document="Рахунок INV-261. Постачальник Vox Alba. До сплати 12,50 UAH.",
        expected={
            "invoice_id": "INV-261",
            "supplier": "Vox Alba",
            "total_minor": 1250,
            "currency": "UAH",
            "due_date": None,
        },
    )


@pytest.mark.parametrize(
    "text",
    [
        '{"invoice_id": null, "invoice_id": "A"}',
        'prefix {"invoice_id": null}',
        '{"invoice_id":null,"supplier":null,"total_minor":false,"currency":null,"due_date":null}',
        '{"invoice_id":null,"supplier":null,"total_minor":NaN,"currency":null,"due_date":null}',
        '{"invoice_id":null,"supplier":null,"total_minor":null,"currency":null,"due_date":"2026-02-30"}',
        '{"invoice_id":null,"supplier":null,"total_minor":null,"currency":null}',
    ],
)
def test_rejects_malformed_and_wrong_types(text: str) -> None:
    assert parse_output("invoice", text)[1] is None


def test_score_checks_absence_and_source(invoice: Case) -> None:
    good = score(invoice, json.dumps(invoice.expected))
    wrong = score(
        invoice, json.dumps({**invoice.expected, "supplier": "Alba Labs", "due_date": "2026-12-01"})
    )
    assert good.total == 1
    assert wrong.unsupported == 2
    assert wrong.abstention == 0
    assert wrong.total < good.total


def test_grounded_distractor_is_still_incorrect(invoice: Case) -> None:
    invoice = invoice.model_copy(update={"document": invoice.document + " Замовник Lumen."})
    result = score(invoice, json.dumps({**invoice.expected, "supplier": "Lumen"}))
    assert result.grounding == 1
    assert result.accuracy < 1
    assert result.total < 1


def test_generated_gold_is_schema_valid_and_supported() -> None:
    for case, _ in synthetic_cases(42):
        scored = score(case, json.dumps(case.expected))
        assert scored.total == 1, case.id


def test_golden_is_balanced_and_auditable(tmp_path: Path) -> None:
    manifest = build_dataset(tmp_path, seed=42)
    golden = load_cases(tmp_path / "golden.jsonl")
    assert [len([c for c in golden if c.language == lang]) for lang in ("uk", "en")] == [150, 150]
    assert manifest["counts"]["golden"] == 300
    audit(
        {name: load_cases(tmp_path / f"{name}.jsonl") for name in ("train", "validation", "golden")}
    )


def test_audit_rejects_leakage(invoice: Case) -> None:
    copied = invoice.model_copy(update={"id": "new", "group": "renamed"})
    with pytest.raises(ValueError, match="Cross-split"):
        audit({"train": [invoice], "golden": [copied]})


def test_preserves_distinct_tasks_on_same_document(invoice: Case) -> None:
    contract = Case(
        id="contract",
        group="manual",
        task="contract",
        language="uk",
        synthetic=True,
        document=invoice.document,
        expected={
            "contract_id": None,
            "counterparty": None,
            "effective_date": None,
            "notice_days": None,
            "jurisdiction": None,
        },
    )
    assert deduplicate([invoice, contract]) == [invoice, contract]
    conflicting = invoice.model_copy(
        update={"id": "other", "expected": {**invoice.expected, "supplier": None}},
    )
    with pytest.raises(ValueError, match="Conflicting"):
        deduplicate([invoice, conflicting])


def test_grounding_pairs_amount_and_currency(invoice: Case) -> None:
    invoice = invoice.model_copy(update={"document": invoice.document + " Deposit 14,50 EUR."})
    prediction = {**invoice.expected, "total_minor": 1450}
    assert score(invoice, json.dumps(prediction)).unsupported == 1

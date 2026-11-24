import json
from pathlib import Path

import pytest

from verifine.data.challenges import add_archival_distractor
from verifine.data.schemas import Case
from verifine.rewards.composite import score
from verifine.train.common import read_config
from verifine.train.run_grpo import make_reward


def test_local_model_override_prevents_an_implicit_download(tmp_path: Path) -> None:
    config = tmp_path / "run.yaml"
    config.write_text("model: Qwen/remote\noutput: artifacts/test", encoding="utf-8")
    settings = read_config(config, str(tmp_path), None, None)
    assert settings.model == str(tmp_path)
    assert not settings.allow_download


def test_trl_reward_supports_conversational_completions(tmp_path: Path) -> None:
    case = Case(
        id="example",
        group="example",
        task="support",
        language="en",
        synthetic=True,
        document="Ticket status for SUP-X100.",
        expected={
            "tool": "lookup_ticket",
            "arguments": {"priority": None, "ticket_id": "SUP-X100"},
        },
    )
    log = tmp_path / "reward.jsonl"
    reward = make_reward(log)
    results = reward(
        completions=[[{"content": json.dumps(case.expected)}], [{"content": "bad"}]],
        case_json=[case.model_dump_json()] * 2,
    )
    assert results == [1, 0]
    assert json.loads(log.read_text().splitlines()[0])["tool_args"] == 1
    challenge = add_archival_distractor(case, appendix=True)
    assert "SUP-OLD-729" in challenge.document
    assert score(challenge, json.dumps(case.expected)).total == 1
    assert (
        score(
            challenge,
            '{"tool":"assign_priority","arguments":{"ticket_id":"SUP-OLD-729","priority":"high"}}',
        ).total
        < 1
    )
    with pytest.raises(ValueError):
        reward(completions=["extra"], case_json=[])

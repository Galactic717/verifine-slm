import json
from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import Any

from verifine.data.challenges import add_archival_distractor
from verifine.data.prompts import messages
from verifine.data.schemas import Case
from verifine.rewards.composite import score
from verifine.train.common import arguments, load_qlora, tracking, trainer_kwargs, verified_splits


def make_reward(log: Path) -> Any:
    def verifiable_reward(
        completions: list[Any],
        case_json: list[str],
        trainer_state: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        results: list[float] = []
        with log.open("a", encoding="utf-8") as handle:
            for completion, payload in zip(completions, case_json, strict=True):
                case = Case.model_validate_json(payload)
                text = completion if isinstance(completion, str) else completion[-1]["content"]
                parts = score(case, text)
                results.append(parts.total)
                handle.write(
                    json.dumps(
                        {
                            "step": getattr(trainer_state, "global_step", None),
                            "id": case.id,
                            "task": case.task,
                            "language": case.language,
                            **asdict(parts),
                        }
                    )
                    + "\n"
                )
        return results

    return verifiable_reward


def main() -> None:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    settings, resume = arguments("GRPO")
    if settings.adapter is None or not (settings.adapter / "adapter_config.json").is_file():
        raise ValueError("Specify a saved SFT adapter with --adapter")
    train, _ = verified_splits(settings)
    model, tokenizer = load_qlora(settings, existing=settings.adapter)
    tokenizer.padding_side = "left"
    rows = []
    rng = Random(settings.seed)
    for case in train:
        if case.synthetic and rng.random() < settings.challenge_ratio:
            case = add_archival_distractor(case, appendix=bool(rng.getrandbits(1)))
        prompt = messages(case.task, case.document)
        tokens = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        length = len(tokens["input_ids"] if hasattr(tokens, "keys") else tokens)
        if length + settings.max_completion_length <= settings.max_length:
            rows.append({"prompt": prompt, "case_json": case.model_dump_json()})
    if not rows:
        raise ValueError("No training prompt fits the GRPO token budget")

    with tracking(settings, "grpo") as callback:
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=Dataset.from_list(rows),
            reward_funcs=make_reward(settings.output / "reward_breakdown.jsonl"),
            callbacks=[callback],
            args=GRPOConfig(
                **trainer_kwargs(settings, evaluation=False),
                num_generations=settings.num_generations,
                generation_batch_size=settings.num_generations,
                max_completion_length=settings.max_completion_length,
                beta=settings.beta,
                use_vllm=False,
                log_completions=False,
                remove_unused_columns=False,
                temperature=0.8,
                scale_rewards="group",
            ),
        )
        trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(str(settings.output))
        tokenizer.save_pretrained(settings.output)
        trainer.state.save_to_json(str(settings.output / "trainer_state.json"))


if __name__ == "__main__":
    main()

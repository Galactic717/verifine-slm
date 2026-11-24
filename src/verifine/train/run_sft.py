import json
from typing import Any

from verifine.data.prompts import messages
from verifine.data.schemas import Case
from verifine.train.common import arguments, load_qlora, tracking, trainer_kwargs, verified_splits


def samples(cases: list[Case], tokenizer: Any, max_length: int) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        prompt = messages(case.task, case.document)
        completion = [
            {
                "role": "assistant",
                "content": json.dumps(
                    case.expected,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        ]
        rendered = tokenizer.apply_chat_template(
            prompt + completion,
            tokenize=True,
            return_dict=True,
        )
        count = len(rendered["input_ids"])
        if count > max_length:
            raise ValueError(f"SFT case {case.id} needs {count} tokens, maximum is {max_length}")
        rows.append({"prompt": prompt, "completion": completion})
    return rows


def main() -> None:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    settings, resume = arguments("SFT")
    train, dev = verified_splits(settings)

    model, tokenizer = load_qlora(settings)
    train_dataset = Dataset.from_list(samples(train, tokenizer, settings.max_length))
    eval_dataset = Dataset.from_list(samples(dev, tokenizer, settings.max_length))
    with tracking(settings, "sft") as callback:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[callback],
            args=SFTConfig(
                **trainer_kwargs(settings, evaluation=True),
                max_length=settings.max_length,
                completion_only_loss=True,
                packing=False,
            ),
        )
        trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(str(settings.output))
        tokenizer.save_pretrained(settings.output)
        trainer.state.save_to_json(str(settings.output / "trainer_state.json"))


if __name__ == "__main__":
    main()

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from verifine.data.split import load_cases


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    revision: str | None = None
    architecture: str = "causal"
    data_dir: Path = Path("data")
    output: Path
    seed: int = 42
    max_steps: int = Field(default=100, ge=1)
    max_length: int = Field(default=1024, ge=256)
    max_completion_length: int = Field(default=160, ge=24)
    gradient_accumulation: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    rank: int = Field(default=16, ge=1)
    adapter: Path | None = None
    allow_download: bool = False
    num_generations: int = Field(default=4, ge=2)
    challenge_ratio: float = Field(default=0, ge=0, le=1)
    beta: float = Field(default=0, ge=0)
    logging_steps: int = Field(default=1, ge=1)
    save_steps: int = Field(default=25, ge=1)
    eval_steps: int = Field(default=25, ge=1)
    mlflow_dir: Path = Path("mlruns")


def read_config(
    path: Path,
    model: str | None,
    output: Path | None,
    steps: int | None,
    allow_download: bool = False,
    adapter: Path | None = None,
    challenge_ratio: float | None = None,
) -> Settings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Training config must be a YAML mapping")
    overrides = {
        "output": output,
        "max_steps": steps,
        "adapter": adapter,
        "challenge_ratio": challenge_ratio,
    }
    raw.update({key: value for key, value in overrides.items() if value is not None})
    if model:
        raw.update(model=model, revision=None)
    raw["allow_download"] = allow_download
    return Settings.model_validate(raw)


def arguments(stage: str) -> tuple[Settings, str | None]:
    parser = argparse.ArgumentParser(description=f"Run {stage} on a CUDA GPU")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", help="Path to an existing Safetensors snapshot")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    if stage == "GRPO":
        parser.add_argument("--adapter", type=Path)
        parser.add_argument("--challenge-ratio", type=float)
    args = parser.parse_args()
    settings = read_config(
        args.config,
        args.model,
        args.output,
        args.max_steps,
        args.allow_download,
        getattr(args, "adapter", None),
        getattr(args, "challenge_ratio", None),
    )
    if (
        settings.output.exists()
        and any(settings.output.iterdir())
        and not args.resume_from_checkpoint
    ):
        parser.error("Use an empty output directory or specify --resume-from-checkpoint")
    return settings, args.resume_from_checkpoint


def load_qlora(
    settings: Settings,
    *,
    existing: Path | None = None,
    inference: bool = False,
) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This profile requires a CUDA GPU with BF16 support")
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("The QLoRA profile expects one process and one CUDA GPU")
    set_seed(settings.seed)
    location = Path(settings.model)
    if location.is_file() or settings.model.lower().endswith(".gguf"):
        raise ValueError("Training needs the original Transformers Safetensors model directory")
    kwargs = {
        "revision": settings.revision,
        "local_files_only": not settings.allow_download,
        "trust_remote_code": False,
    }
    if settings.architecture not in {"causal", "gemma4"}:
        raise ValueError("Architecture must be causal or gemma4")
    loader = (
        AutoModelForImageTextToText if settings.architecture == "gemma4" else AutoModelForCausalLM
    )
    tokenizer = AutoTokenizer.from_pretrained(
        existing or settings.model, **({"local_files_only": True} if existing else kwargs)
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = loader.from_pretrained(
        settings.model,
        **kwargs,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    model.config.use_cache = inference
    if not inference:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    if existing:
        model = PeftModel.from_pretrained(model, str(existing), is_trainable=not inference)
        # The serialized adapter must remain portable if the base path was supplied explicitly.
        if settings.revision:
            model.peft_config["default"].revision = settings.revision
    elif not inference:
        model = get_peft_model(
            model,
            LoraConfig(
                r=settings.rank,
                lora_alpha=2 * settings.rank,
                lora_dropout=0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
                revision=settings.revision,
            ),
        )
    return model, tokenizer


def verified_splits(settings: Settings) -> tuple[list[Any], list[Any]]:
    from verifine.data.split import audit

    train = load_cases(settings.data_dir / "train.jsonl")
    validation = load_cases(settings.data_dir / "validation.jsonl")
    if not train or not validation:
        raise ValueError("Training and validation must be non-empty")
    audit({"train": train, "validation": validation})
    return train, validation


def trainer_kwargs(settings: Settings, *, evaluation: bool) -> dict[str, Any]:
    return {
        "output_dir": str(settings.output),
        "max_steps": settings.max_steps,
        "seed": settings.seed,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": settings.gradient_accumulation,
        "learning_rate": settings.learning_rate,
        "warmup_ratio": 0.06,
        "optim": "paged_adamw_8bit",
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": settings.logging_steps,
        "report_to": "none",
        "save_strategy": "steps",
        "save_steps": settings.save_steps,
        "save_total_limit": 2,
        "eval_strategy": "steps" if evaluation else "no",
        "eval_steps": settings.eval_steps,
    }


@contextmanager
def tracking(settings: Settings, stage: str) -> Any:
    import mlflow
    import torch
    from transformers import TrainerCallback

    class LogCallback(TrainerCallback):
        def on_log(
            self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any
        ) -> None:
            if logs:
                mlflow.log_metrics(
                    {
                        key: float(value)
                        for key, value in logs.items()
                        if isinstance(value, int | float)
                    },
                    step=state.global_step,
                )

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(settings.mlflow_dir.resolve().as_uri())
    mlflow.set_experiment("verifine-slm")
    settings.output.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name=stage):
        config = settings.model_dump(mode="json")
        for key, value in config.items():
            if value is not None:
                mlflow.log_param(key, value)
        mlflow.log_param("device", torch.cuda.get_device_name(0))
        mlflow.log_param("torch", torch.__version__)
        mlflow.log_param("stage", stage)
        (settings.output / "run_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        yield LogCallback()
        peak_mib = torch.cuda.max_memory_reserved() / (1024**2)
        mlflow.log_metric("peak_cuda_reserved_mib", peak_mib)
        (settings.output / "hardware.json").write_text(
            json.dumps(
                {"gpu": torch.cuda.get_device_name(0), "peak_cuda_reserved_mib": peak_mib}, indent=2
            ),
            encoding="utf-8",
        )

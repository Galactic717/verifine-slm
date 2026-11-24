#!/usr/bin/env bash
set -euo pipefail

train_python="${VERIFINE_TRAIN_PYTHON:-$HOME/.local/share/verifine-train/bin/python}"
out="${1:-artifacts/eval/qwen-ablation}"
sft="${VERIFINE_SFT_ADAPTER:-artifacts/qwen-sft}"
grpo="${VERIFINE_GRPO_ADAPTER:-artifacts/qwen-grpo}"
model_args=()
if [[ -n "${VERIFINE_QWEN_MODEL:-}" ]]; then
  if [[ ! -f "$VERIFINE_QWEN_MODEL/config.json" ]]; then
    printf 'Set VERIFINE_QWEN_MODEL to an existing Qwen Safetensors snapshot.\n' >&2
    exit 2
  fi
  model_args=(--model "$VERIFINE_QWEN_MODEL")
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
test -f "$sft/adapter_config.json"
test -f "$grpo/adapter_config.json"
"$train_python" -m verifine.train.run_hf_eval --config configs/qwen3_4b_sft.yaml \
  "${model_args[@]}" --label "Qwen3-4B" --output "$out" \
  --batch-size "${VERIFINE_EVAL_BATCH_SIZE:-8}" --ablation "$sft" "$grpo"
"$train_python" -m verifine.cli compare "$out/base.json" "$out/sft.json" --output "$out/base-vs-sft.json"
"$train_python" -m verifine.cli compare "$out/sft.json" "$out/grpo.json" --output "$out/sft-vs-grpo.json"

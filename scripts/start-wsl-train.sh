#!/usr/bin/env bash
set -euo pipefail

stage="${1:?Choose sft or grpo as the first argument}"
shift
case "$stage" in
  sft|grpo) ;;
  *) printf 'Unknown training stage %s\n' "$stage" >&2; exit 2 ;;
esac

train_python="${VERIFINE_TRAIN_PYTHON:-$HOME/.local/share/verifine-train/bin/python}"
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
exec "$train_python" -m "verifine.train.run_$stage" --config "configs/qwen3_4b_$stage.yaml" \
  "${model_args[@]}" "$@"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.conda/smolvla_lebai/bin/python"
export HF_HOME="$ROOT/.cache/huggingface"
STEPS="${1:-20}"
if [[ "$STEPS" == 20 ]]; then
  OUT="outputs/train/smolvla_lebai_smoke20_run2"
else
  OUT="outputs/train/smolvla_lebai_${STEPS}"
fi
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing run: $OUT" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUT")"
if [[ -e "$OUT.log" ]]; then
  echo "Refusing to overwrite existing log: $OUT.log" >&2
  exit 2
fi
"$PY" -m lerobot.scripts.lerobot_train \
  --config_path=smolvla_lebai/configs/train.yaml \
  --steps="$STEPS" --output_dir="$OUT" \
  --save_freq="$([[ "$STEPS" == 20 ]] && echo 20 || echo 200)" \
  2>&1 | tee "$OUT.log"

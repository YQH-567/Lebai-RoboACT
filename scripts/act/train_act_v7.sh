#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -e outputs/train/act_lebai_lm3_v7 ]]; then
    echo "Refusing to overwrite outputs/train/act_lebai_lm3_v7" >&2
    exit 1
fi

python scripts/act/validate_act_v7_dataset.py

if ! python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA is required for the 60000-step ACT v7 training run." >&2
    exit 1
fi

lerobot-train \
    --dataset.repo_id=local/lebai_lm3_pick_place_v7_train \
    --dataset.root=datasets/lerobot/lebai_lm3_pick_place_v7/train \
    --dataset.eval_split=0.0 \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.vision_backbone=resnet18 \
    --policy.chunk_size=30 \
    --policy.n_action_steps=10 \
    --batch_size=8 \
    --steps=60000 \
    --eval_steps=0 \
    --save_freq=20000 \
    --output_dir=outputs/train/act_lebai_lm3_v7 \
    --job_name=act_lebai_lm3_v7

python scripts/act/validate_act_v7_model.py \
    --checkpoint outputs/train/act_lebai_lm3_v7/checkpoints/060000/pretrained_model

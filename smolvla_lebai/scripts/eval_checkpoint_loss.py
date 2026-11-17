#!/usr/bin/env python3
"""Forward-only, phase-balanced train/validation loss on saved SmolVLA checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def split_indices(split: str, episodes: int):
    manifest = json.loads((ROOT / "datasets/lerobot/lebai_lm3_pick_place_25hz/split_manifest.json").read_text())
    source_ids = manifest[f"{split}_dataset_episode_to_source"]
    for ep in range(min(episodes, len(source_ids))):
        with h5py.File(ROOT / f"datasets/raw_25hz/episode_{source_ids[ep]:06d}.hdf5", "r") as h5:
            fsm = h5["fsm_state"][:]
        for stage in range(9):
            matches = np.flatnonzero(fsm == stage)
            if len(matches):
                yield ep, stage, int(matches[len(matches) // 2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-split", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/eval/smolvla_diagnostics/checkpoint_loss.json")
    args = parser.parse_args()
    checkpoints = [ROOT / f"outputs/train/smolvla_lebai_1000/checkpoints/{step:06d}/pretrained_model"
                   for step in (200, 1000)]
    datasets = {}
    for split in ("train", "val"):
        datasets[split] = LeRobotDataset(
            repo_id=f"local/lebai_lm3_pick_place_25hz_smolvla31_{split}",
            root=ROOT / f"datasets/lerobot/lebai_lm3_pick_place_25hz_smolvla31/{split}",
            video_backend="pyav", delta_timestamps={"action": [i / 25 for i in range(50)]},
        )
    result = {"method": "Forward only; fixed phase-midpoint observations and per-sample RNG seeds; 50-step action supervision."}
    for checkpoint in checkpoints:
        model = SmolVLAPolicy.from_pretrained(checkpoint).to("cuda").eval()
        pre, _ = make_pre_post_processors(policy_cfg=model.config, pretrained_path=checkpoint,
                                        preprocessor_overrides={"device_processor": {"device": "cuda"}})
        point = {}
        for split, dataset in datasets.items():
            losses, phases = [], {stage: [] for stage in range(9)}
            starts = np.asarray(dataset.meta.episodes["dataset_from_index"])
            for ep, stage, frame in split_indices(split, args.episodes_per_split):
                sample = dataset[int(starts[ep]) + frame]
                batch = {key: sample[key] for key in (
                    "observation.state", "observation.images.front", "observation.images.wrist",
                    "task", "action", "action_is_pad")}
                # The inference preprocessor batches observations; training DataLoader
                # already batches the future-action tensors before calling it.
                batch["action"] = batch["action"].unsqueeze(0)
                batch["action_is_pad"] = batch["action_is_pad"].unsqueeze(0)
                pre.reset()
                batch = pre(batch)
                seed = 314159 + ep * 100 + stage
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                with torch.inference_mode():
                    loss, _ = model(batch)
                value = float(loss.detach().cpu())
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite loss: {split}, episode {ep}, stage {stage}")
                losses.append(value); phases[stage].append(value)
            point[split] = {"n": len(losses), "mean_loss": float(np.mean(losses)),
                            "stage_mean_loss": {str(stage): float(np.mean(v)) for stage, v in phases.items() if v}}
        result[checkpoint.parent.name] = point
        del model, pre
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

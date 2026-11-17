#!/usr/bin/env python3
"""Reconstruct the first 1000 LeRobot sampler indices from the saved seed/config."""
from __future__ import annotations

import itertools
import json
import os
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.sampler import EpisodeAwareSampler

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "datasets/lerobot/lebai_lm3_pick_place_25hz_smolvla31/train"
SEED = 20261117
STEPS = 1000
CHUNK = 50


def main():
    config = json.loads((ROOT / "outputs/train/smolvla_lebai_1000/checkpoints/001000/pretrained_model/train_config.json").read_text())
    assert config["seed"] == SEED and config["steps"] == STEPS and config["batch_size"] == 1
    assert config["policy"]["chunk_size"] == CHUNK
    ep_table = pq.read_table(BASE / "meta/episodes/chunk-000/file-000.parquet",
                             columns=["dataset_from_index", "dataset_to_index"])
    starts = np.asarray(ep_table["dataset_from_index"], dtype=int)
    ends = np.asarray(ep_table["dataset_to_index"], dtype=int)
    sampler = EpisodeAwareSampler(starts, ends, shuffle=True, seed=SEED)
    sampled = np.fromiter(itertools.islice(iter(sampler), STEPS), dtype=int)
    episodes = np.searchsorted(ends, sampled, side="right")
    frames = sampled - starts[episodes]
    manifest = json.loads((ROOT / "datasets/lerobot/lebai_lm3_pick_place_25hz/split_manifest.json").read_text())
    sample_stages, full_stages = Counter(), Counter()
    supervised = set()
    for ep, source_id in enumerate(manifest["train_dataset_episode_to_source"]):
        with h5py.File(ROOT / f"datasets/raw_25hz/episode_{source_id:06d}.hdf5", "r") as h5:
            fsm = h5["fsm_state"][:].astype(int)
        full_stages.update(fsm.tolist())
        for frame in frames[episodes == ep]:
            sample_stages[int(fsm[frame])] += 1
            supervised.update((ep, step) for step in range(frame, min(frame + CHUNK, len(fsm))))
    result = {
        "method": "Reconstructed from LeRobot 0.7.0 EpisodeAwareSampler(seed, epoch=0); sample indices were not logged during training.",
        "seed": SEED, "optimizer_steps": STEPS, "batch_size": 1, "chunk_size": CHUNK,
        "distinct_observation_frames": int(len(np.unique(sampled))),
        "distinct_episodes": int(len(np.unique(episodes))),
        "samples_per_episode_min": int(min(Counter(episodes).values())),
        "samples_per_episode_max": int(max(Counter(episodes).values())),
        "nominal_future_action_targets": STEPS * CHUNK,
        "unique_future_action_labels": len(supervised),
        "unique_future_action_label_fraction": len(supervised) / int(sum(ends - starts)),
        "sample_fsm_counts": dict(sorted(sample_stages.items())),
        "full_train_fsm_counts": dict(sorted(full_stages.items())),
    }
    out = ROOT / "outputs/eval/smolvla_diagnostics/training_coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

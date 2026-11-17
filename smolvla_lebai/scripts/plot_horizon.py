#!/usr/bin/env python3
"""Plot measured expert FSM timing and paired 95/240-step SmolVLA diagnostics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "smolvla_lebai/figures"
FIG.mkdir(parents=True, exist_ok=True)
NAMES = ["Rest", "Above", "Approach", "Grasp", "Lift", "Place", "Release", "Retreat", "Done"]


def main():
    manifest = json.loads((ROOT / "datasets/lerobot/lebai_lm3_pick_place_25hz/split_manifest.json").read_text())
    sources = manifest["train_source_episode_ids"] + manifest["val_source_episode_ids"]
    first = {stage: [] for stage in range(9)}
    for source in sources:
        with h5py.File(ROOT / f"datasets/raw_25hz/episode_{source:06d}.hdf5", "r") as h5:
            fsm = h5["fsm_state"][:]
        for stage in range(9):
            matches = np.flatnonzero(fsm == stage)
            if len(matches):
                first[stage].append(matches[0])
    fig, ax = plt.subplots(figsize=(9, 4))
    med = [np.median(first[s]) for s in range(9)]
    lo = [np.quantile(first[s], .1) for s in range(9)]
    hi = [np.quantile(first[s], .9) for s in range(9)]
    ax.errorbar(range(9), med, yerr=[np.array(med) - lo, np.array(hi) - med], fmt="o", capsize=4)
    ax.axhline(95, color="red", linestyle="--", label="95-step evaluation cutoff")
    ax.set_xticks(range(9), NAMES, rotation=30); ax.set_ylabel("First frame of stage (25 Hz)")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(FIG / "expert_fsm_timing.png", dpi=150); plt.close(fig)

    base = ROOT / "outputs/eval"
    with (base / "smolvla_h240_seed20261118_retry/steps.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for episode in (1, 2, 3):
        records = [row for row in rows if int(row["episode"]) == episode]
        t = np.array([int(row["policy_step"]) for row in records])
        obj_z = np.array([json.loads(row["object_position"])[2] for row in records])
        distance = np.array([float(row["ee_object_center_distance"]) for row in records])
        grip = np.array([float(row["gripper"]) for row in records])
        axes[0].plot(t, obj_z, label=f"episode {episode}")
        axes[1].plot(t, distance, label=f"episode {episode}")
        axes[2].step(t, grip + (episode - 1) * .12, where="post", label=f"episode {episode}")
    for ax in axes:
        ax.axvline(95, color="red", linestyle="--", alpha=.7); ax.grid(alpha=.2)
    axes[0].axhline(.1, color="gray", linestyle=":", alpha=.6)
    axes[0].set_ylabel("Object Z (m)"); axes[0].legend()
    axes[1].set_ylabel("EE-object distance (m)")
    axes[2].set_ylabel("Gripper sign (+ offset)"); axes[2].set_xlabel("Policy step (25 Hz)")
    fig.tight_layout(); fig.savefig(FIG / "extended_horizon_three_episodes.png", dpi=150); plt.close(fig)
    print(FIG / "expert_fsm_timing.png", FIG / "extended_horizon_three_episodes.png")


if __name__ == "__main__":
    main()

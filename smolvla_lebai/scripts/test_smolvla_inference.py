#!/usr/bin/env python3
"""Offline SmolVLA inference on held-out 25Hz expert observations; no simulator feedback."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
DATASET_ROOT = ROOT / "datasets/lerobot/lebai_lm3_pick_place_25hz_smolvla31"


def locate_checkpoint(path: Path) -> Path:
    if (path / "model.safetensors").is_file():
        return path
    candidates = sorted(path.glob("checkpoints/*/pretrained_model"))
    if not candidates:
        raise FileNotFoundError(f"No saved checkpoint under {path}")
    return candidates[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/train/smolvla_lebai_1000")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--frames-per-episode", type=int, default=10)
    p.add_argument("--fsm-windows", action="store_true", help="Sample one contiguous window per FSM stage using source HDF5 labels.")
    p.add_argument("--transition-windows", action="store_true", help="Sample 10 frames before and after grasp/release transitions.")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    checkpoint = locate_checkpoint(args.checkpoint)
    dataset = LeRobotDataset(repo_id=f"local/lebai_lm3_pick_place_25hz_smolvla31_{args.split}",
                             root=DATASET_ROOT / args.split, video_backend="pyav")
    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.to("cuda").eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    assert policy.config.input_features["observation.state"].shape == (31,)
    assert policy.config.output_features["action"].shape == (8,)
    assert len([k for k in policy.config.input_features if k.startswith("observation.images.")]) == 2
    assert policy.config.max_state_dim == 32 and policy.config.max_action_dim == 32
    max_episode = min(args.episodes, dataset.meta.total_episodes)
    indices = dataset.hf_dataset["episode_index"]
    manifest = json.loads((ROOT / "datasets/lerobot/lebai_lm3_pick_place_25hz/split_manifest.json").read_text())
    source_ids = manifest[f"{args.split}_dataset_episode_to_source"]
    rows, latencies = [], []
    for episode in range(max_episode):
        episode_indices = np.flatnonzero(np.asarray(indices) == episode)
        with h5py.File(ROOT / f"datasets/raw_25hz/episode_{source_ids[episode]:06d}.hdf5", "r") as h5:
            fsm = h5["fsm_state"][:].astype(int)
        windows = []
        if args.fsm_windows and args.transition_windows:
            raise ValueError("Choose only one diagnostic window mode")
        if args.transition_windows:
            for stage in (3, 6):
                changes = np.flatnonzero(fsm == stage)
                if not len(changes):
                    continue
                start = max(0, int(changes[0]) - 10)
                windows.append((stage, episode_indices[start : start + 20].tolist()))
        elif args.fsm_windows:
            for stage in range(9):
                stage_indices = np.flatnonzero(fsm == stage)
                if not len(stage_indices):
                    continue
                window = min(args.frames_per_episode, len(stage_indices))
                start = int(stage_indices[len(stage_indices) // 2] - window // 2)
                start = max(int(stage_indices[0]), min(start, int(stage_indices[-1] - window + 1)))
                windows.append((stage, episode_indices[start : start + window].tolist()))
        else:
            # Three sequential windows: beginning, middle, end. Use --fsm-windows for full phase coverage.
            window = max(1, args.frames_per_episode // 3)
            starts = [0, max(0, len(episode_indices) // 2 - window // 2), max(0, len(episode_indices) - window)]
            windows = [(int(fsm[start]), episode_indices[start : start + window].tolist()) for start in starts]
        for stage, segment_indices in windows:
            policy.reset(); preprocessor.reset(); postprocessor.reset()
            for chunk_offset, index in enumerate(segment_indices):
                sample = dataset[index]
                observation = {key: sample[key] for key in (
                    "observation.state", "observation.images.front", "observation.images.wrist", "task"
                )}
                t0 = time.perf_counter()
                with torch.inference_mode():
                    prediction = postprocessor(policy.select_action(preprocessor(observation)))
                    torch.cuda.synchronize()
                latency = time.perf_counter() - t0
                pred = prediction.detach().float().cpu().numpy().reshape(-1)
                expert = sample["action"].numpy()
                assert pred.shape == expert.shape == (8,) and np.isfinite(pred).all()
                norm = float(np.linalg.norm(pred[3:7]))
                quat = pred[3:7] / max(norm, 1e-8)
                angle = 2 * np.degrees(np.arccos(np.clip(abs(float(np.dot(quat, expert[3:7]))), 0, 1)))
                rows.append({"episode": episode, "frame": int(sample["frame_index"]),
                             "fsm_state": int(fsm[int(sample["frame_index"])]), "window_stage": stage,
                             "chunk_offset": chunk_offset,
                             "predicted": pred.tolist(), "expert": expert.tolist(),
                             "xyz_mae_m": float(np.mean(np.abs(pred[:3] - expert[:3]))),
                             "quat_angle_deg": float(angle), "quat_norm": norm,
                             "gripper_correct": bool((pred[7] >= 0) == (expert[7] >= 0)),
                             "latency_s": latency})
                latencies.append(latency)
    result = {"checkpoint": str(checkpoint.resolve()), "split": args.split,
              "fsm_windows": args.fsm_windows, "transition_windows": args.transition_windows,
              "frames": len(rows), "episodes": max_episode,
              "chunk_size": policy.config.chunk_size, "n_action_steps": policy.config.n_action_steps,
              "control_period_s": 1 / dataset.fps,
              "xyz_mae_m": float(np.mean([x["xyz_mae_m"] for x in rows])),
              "quat_angle_mean_deg": float(np.mean([x["quat_angle_deg"] for x in rows])),
              "gripper_accuracy": float(np.mean([x["gripper_correct"] for x in rows])),
              "latency_mean_s": float(np.mean(latencies)), "latency_p95_s": float(np.quantile(latencies, .95)),
              "max_quat_norm_error": float(max(abs(x["quat_norm"] - 1) for x in rows)),
              "examples": rows[:5]}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({**result, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()

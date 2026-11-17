#!/usr/bin/env python3
"""Validate the original and SmolVLA 31D LeRobot v3 train/val datasets."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
BASE = ROOT / "datasets/lerobot/lebai_lm3_pick_place_25hz"
DERIVED = ROOT / "datasets/lerobot/lebai_lm3_pick_place_25hz_smolvla31"
ACTION_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]


def files(root: Path, pattern: str):
    return sorted(root.glob(pattern))


def validate_split(split: str) -> dict:
    source, target = BASE / split, DERIVED / split
    src_info = json.loads((source / "meta/info.json").read_text())
    info = json.loads((target / "meta/info.json").read_text())
    assert info["features"]["observation.state"]["names"] == src_info["features"]["observation.state"]["names"][:31]
    assert info["features"]["observation.state"]["shape"] == [31]
    assert info["features"]["action"]["names"] == ACTION_NAMES
    assert info["features"]["action"] == src_info["features"]["action"]
    assert info["fps"] == 25 and info["total_tasks"] == 1
    stats = json.loads((target / "meta/stats.json").read_text())
    src_stats = json.loads((source / "meta/stats.json").read_text())
    for key, value in stats["observation.state"].items():
        assert value == (src_stats["observation.state"][key][:31] if key != "count" else src_stats["observation.state"][key])
    for key in ("observation.images.front", "observation.images.wrist", "action"):
        assert stats[key] == src_stats[key]
    src_files, dst_files = files(source, "data/**/*.parquet"), files(target, "data/**/*.parquet")
    assert len(src_files) == len(dst_files)
    all_state, all_action, all_episode, all_frame, all_time = [], [], [], [], []
    for p, q in zip(src_files, dst_files):
        a, b = pq.read_table(p), pq.read_table(q)
        state_a = np.asarray(a["observation.state"].to_pylist(), dtype=np.float32)
        state_b = np.asarray(b["observation.state"].to_pylist(), dtype=np.float32)
        assert np.array_equal(state_a[:, :31], state_b)
        for key in ("action", "timestamp", "frame_index", "episode_index", "index", "task_index"):
            assert a[key].equals(b[key]), key
        all_state.append(state_b)
        all_action.append(np.asarray(b["action"].to_pylist(), dtype=np.float32))
        all_episode.append(np.asarray(b["episode_index"].to_pylist()))
        all_frame.append(np.asarray(b["frame_index"].to_pylist()))
        all_time.append(np.asarray(b["timestamp"].to_pylist()))
    state = np.concatenate(all_state)
    action = np.concatenate(all_action)
    episode = np.concatenate(all_episode)
    frame = np.concatenate(all_frame)
    timestamp = np.concatenate(all_time)
    assert np.isfinite(state).all() and np.isfinite(action).all()
    assert len(state) == info["total_frames"]
    assert len(set(episode)) == info["total_episodes"]
    for ep in np.unique(episode):
        mask = episode == ep
        assert np.array_equal(frame[mask], np.arange(mask.sum()))
        assert np.allclose(np.diff(timestamp[mask]), 0.04, atol=1e-5)
    assert np.allclose(np.linalg.norm(action[:, 3:7], axis=1), 1.0, atol=0.03)
    assert set(np.unique(action[:, 7]).tolist()) <= {-1.0, 1.0}
    for camera in ("front", "wrist"):
        src_videos = files(source, f"videos/observation.images.{camera}/**/*.mp4")
        dst_videos = files(target, f"videos/observation.images.{camera}/**/*.mp4")
        assert len(src_videos) == len(dst_videos) and len(src_videos) > 0
        assert all(p.stat().st_ino == q.stat().st_ino for p, q in zip(src_videos, dst_videos))
    dataset = LeRobotDataset(repo_id=f"local/lebai_lm3_pick_place_25hz_smolvla31_{split}", root=target, video_backend="pyav")
    assert len(dataset) == len(state)
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        sample = dataset[index]
        assert sample["observation.state"].shape == (31,)
        assert sample["action"].shape == (8,)
        assert sample["task"] == "Pick up the cube and place it into the tray."
        for camera in ("front", "wrist"):
            image = sample[f"observation.images.{camera}"]
            assert image.shape == (3, 180, 320) and np.isfinite(image.numpy()).all()
    jumps = np.linalg.norm(np.diff(action[:, :3], axis=0), axis=1)
    jumps = jumps[episode[1:] == episode[:-1]]
    return {"episodes": int(info["total_episodes"]), "frames": len(state), "max_action_xyz_step_m": float(jumps.max()), "p99_action_xyz_step_m": float(np.quantile(jumps, 0.99)), "episode_ids": set(episode.tolist())}


def main():
    train, val = validate_split("train"), validate_split("val")
    manifest = json.loads((BASE / "split_manifest.json").read_text())
    train_source = set(manifest["train_source_episode_ids"])
    val_source = set(manifest["val_source_episode_ids"])
    assert not train_source & val_source
    assert len(train_source) == train["episodes"] and len(val_source) == val["episodes"]
    for result in (train, val):
        result.pop("episode_ids")
    print(json.dumps({"train": train, "val": val, "source_episode_overlap": 0}, indent=2))


if __name__ == "__main__":
    main()

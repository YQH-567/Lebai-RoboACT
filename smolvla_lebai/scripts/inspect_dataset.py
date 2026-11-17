#!/usr/bin/env python3
"""Inspect the source 25Hz LeRobot v3 dataset with the official API."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
BASE = ROOT / "datasets/lerobot/lebai_lm3_pick_place_25hz"


def main() -> None:
    manifest = json.loads((BASE / "split_manifest.json").read_text())
    assert manifest["selection_rule"] == "success == 1 AND fsm_done == True"
    assert all(ep["success"] == 1 and ep["fsm_done"] for ep in manifest["scanned_episodes"])
    for split in ("train", "val"):
        data = LeRobotDataset(repo_id=f"local/lebai_lm3_pick_place_25hz_{split}",
                              root=BASE / split, video_backend="pyav")
        meta = data.meta
        sample = data[0]
        assert sample["observation.state"].shape == (34,)
        assert sample["action"].shape == (8,)
        for camera in meta.camera_keys:
            assert sample[camera].shape == (3, 180, 320)
            assert np.isfinite(sample[camera].numpy()).all()
        print(json.dumps({"split": split, "episodes": meta.total_episodes,
                          "frames": len(data), "fps": data.fps,
                          "task": sample["task"],
                          "state_names": meta.features["observation.state"]["names"],
                          "action_names": meta.features["action"]["names"],
                          "camera_keys": meta.camera_keys}, indent=2))


if __name__ == "__main__":
    main()

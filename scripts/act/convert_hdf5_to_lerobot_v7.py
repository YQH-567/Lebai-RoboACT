"""Convert 200 FSM-complete HDF5 episodes into isolated ACT v7 train/val datasets."""
"""
这里转化了hdf5的坐标位置信息，速度，eepose,物体相对位置，相机啥的，对应实验3
"""


import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from act_v7_state import build_state, build_state_names
from convert_hdf5_to_lerobot import ACTION_NAMES, TASK, validate_episode


SEED = 20261116
TRAIN_EPISODES = 160
VAL_EPISODES = 40


def source_episode_id(path):
    try:
        return int(path.stem.removeprefix("episode_"))
    except ValueError as exc:
        raise ValueError(f"Invalid episode filename: {path.name}") from exc


def read_episode(path, expected_fps=None):
    with h5py.File(path, "r") as f:
        n, fps, joint_names, action, qpos, qvel, ee_pose, wrist, front = validate_episode(
            f, path, expected_fps=expected_fps
        )
        # The existing recorder stores this at the HDF5 root, not under observations/.
        if "object_position" not in f:
            raise KeyError(f"{path.name}: missing root-level object_position")
        object_position = f["object_position"][:]
        if object_position.shape != (n, 3) or not np.isfinite(object_position).all():
            raise ValueError(f"{path.name}: invalid object_position {object_position.shape}")
        states = np.stack(
            [build_state(qpos[i], qvel[i], ee_pose[i], object_position[i]) for i in range(n)]
        )
        if states.shape != (n, 34) or not np.isfinite(states).all():
            raise ValueError(f"{path.name}: invalid 34D observation.state")
        yield fps, joint_names, action, states, wrist, front


def make_features(joint_names):
    return {
        "observation.state": {
            "dtype": "float32", "shape": (34,), "names": build_state_names(joint_names),
        },
        "action": {"dtype": "float32", "shape": (8,), "names": ACTION_NAMES},
        "observation.images.wrist": {
            "dtype": "video", "shape": (180, 320, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.front": {
            "dtype": "video", "shape": (180, 320, 3),
            "names": ["height", "width", "channels"],
        },
    }


def convert_split(paths, split, output_root, fps, joint_names):
    dataset = LeRobotDataset.create(
        repo_id=f"local/lebai_lm3_pick_place_v7_{split}",
        root=output_root / split,
        fps=fps,
        robot_type="lebai_lm3_lmg90",
        features=make_features(joint_names),
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=2,
        batch_encoding_size=1,
    )
    try:
        for dataset_ep, path in enumerate(paths):
            for ep_fps, names, action, states, wrist, front in read_episode(path, fps):
                if ep_fps != fps or names != joint_names:
                    raise ValueError(f"{path.name}: FPS or joint order changed")
                for i in range(len(action)):
                    dataset.add_frame({
                        "observation.state": states[i],
                        "action": np.asarray(action[i], dtype=np.float32),
                        "observation.images.wrist": np.ascontiguousarray(wrist[i]),
                        "observation.images.front": np.ascontiguousarray(front[i]),
                        "task": TASK,
                    })
                dataset.save_episode(parallel_encoding=False)
            print(f"[{split}] {dataset_ep + 1}/{len(paths)} {path.name}", flush=True)
    except Exception:
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer(delete_images=True)
        raise
    finally:
        dataset.finalize()
    if dataset.meta.total_episodes != len(paths):
        raise RuntimeError(f"{split}: converted episode count mismatch")
    return dataset.meta.total_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/raw"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("datasets/lerobot/lebai_lm3_pick_place_v7"),
    )
    args = parser.parse_args()
    files = sorted(args.input_dir.glob("episode_*.hdf5"))
    ids = [source_episode_id(path) for path in files]
    if len(files) != TRAIN_EPISODES + VAL_EPISODES or len(set(ids)) != len(ids):
        raise ValueError("Expected exactly 200 uniquely numbered HDF5 episodes")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")

    # Preflight every episode before creating either destination dataset.
    fps = None
    joint_names = None
    for path in files:
        for ep_fps, names, _, _, _, _ in read_episode(path, fps):
            if joint_names is None:
                fps, joint_names = ep_fps, names
            elif names != joint_names:
                raise ValueError(f"{path.name}: joint order changed")
    if fps != 10:
        raise ValueError(f"Expected 10 Hz demonstrations, got {fps}")

    permutation = np.random.default_rng(SEED).permutation(len(files))
    train_paths = [files[i] for i in permutation[:TRAIN_EPISODES]]
    val_paths = [files[i] for i in permutation[TRAIN_EPISODES:]]
    train_ids = [source_episode_id(path) for path in train_paths]
    val_ids = [source_episode_id(path) for path in val_paths]
    if set(train_ids) & set(val_ids):
        raise RuntimeError("Train/validation episode overlap")

    train_frames = convert_split(train_paths, "train", args.output_dir, fps, joint_names)
    val_frames = convert_split(val_paths, "val", args.output_dir, fps, joint_names)
    manifest = {
        "seed": SEED,
        "source_dir": str(args.input_dir.resolve()),
        "object_position_hdf5_key": "object_position",
        "train_source_episode_ids": train_ids,
        "val_source_episode_ids": val_ids,
        "train_dataset_episode_to_source": train_ids,
        "val_dataset_episode_to_source": val_ids,
        "train_frames": train_frames,
        "val_frames": val_frames,
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}: 160 train / 40 val episodes")


if __name__ == "__main__":
    main()

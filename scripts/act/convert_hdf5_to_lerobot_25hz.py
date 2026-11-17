#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lebai LM3 + LMG90：25Hz RGB-D HDF5 → LeRobot Dataset（训练/验证）

相对 convert_hdf5_to_lerobot_v7.py 的主要改动：
1. 强制核对真实 fps=25、timestamp 间隔=0.04s，不再沿用原版 10Hz 校验。
2. 兼容新记录器：读取 HDF5 根目录的 success / fsm_done，以及
   observations/depth/{wrist,front}，不把 FSM 完成当作抓取放置成功。
3. 默认只使用 success=1 且 fsm_done=True 的示教；失败/未知轨迹不进入
   行为克隆训练。默认目标 160 train + 40 val，按 Episode 划分。
4. 保持 ACT v7 的 34D state：qpos12 + qvel12 + ee_pose7 + (object-ee)3；
   8D 绝对位姿动作不变；双 RGB 不变。
5. 深度默认仅校验、保留在原始 HDF5（无损原始 float32）；带 --include-depth
   可以另外生成带两路 LeRobot 深度视频的 RGB-D 数据集。注意：这种数据集
   不能不改视觉编码器就直接交给原来只吃 RGB 的 ACT。
6. 输出独立目录和 split_manifest.json（含数据源 ID / 成功标签 / 深度键）。
7. 提供 --scan-only 仅扫描，不需要导入 LeRobot，用于采集前检查。

使用：
    conda activate env_act
    python scripts/act/convert_hdf5_to_lerobot_25hz.py --scan-only
    python scripts/act/convert_hdf5_to_lerobot_25hz.py

若实际成功演示少于200条：
    python scripts/act/convert_hdf5_to_lerobot_25hz.py --allow-fewer-successes

需要可直接读取深度的 LeRobot 副本时（不要覆盖 RGB-only 副本）：
    python scripts/act/convert_hdf5_to_lerobot_25hz.py --include-depth \
      --output-dir datasets/lerobot/lebai_lm3_pick_place_25hz_rgbd \
      --repo-prefix local/lebai_lm3_pick_place_25hz_rgbd
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

FPS = 25
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 320
SEED = 20261116
TASK = "Pick up the cube and place it into the tray."
ACTION_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]

RGB_KEYS = {
    "wrist": "observations/images/wrist",
    "front": "observations/images/front",
}
DEPTH_KEYS = {
    "wrist": "observations/depth/wrist",
    "front": "observations/depth/front",
}


def decode_attr(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def build_state_names(joint_names):
    return (
        [f"qpos_{name}" for name in joint_names]
        + [f"qvel_{name}" for name in joint_names]
        + ["ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz"]
        + ["object_rel_x", "object_rel_y", "object_rel_z"]
    )


def build_state(qpos, qvel, ee_pose, object_position):
    """环境局部坐标系：object-relative 的定义与 ACT v7 一致。"""
    state = np.concatenate(
        [
            np.asarray(qpos, dtype=np.float32),
            np.asarray(qvel, dtype=np.float32),
            np.asarray(ee_pose, dtype=np.float32),
            np.asarray(object_position, dtype=np.float32)
            - np.asarray(ee_pose[:3], dtype=np.float32),
        ],
        axis=0,
    )
    if state.shape != (34,) or not np.isfinite(state).all():
        raise ValueError(f"Expected 34 finite state entries; got {state.shape}")
    return state


def episode_id(path):
    try:
        return int(path.stem.removeprefix("episode_"))
    except ValueError as exc:
        raise ValueError(f"Bad episode file name: {path}") from exc


@dataclass(frozen=True)
class Episode:
    path: Path
    source_id: int
    success: int
    fsm_done: bool
    frames: int
    fps: int
    joint_names: tuple[str, ...]
    reason: str


def scan_metadata(path: Path) -> Episode:
    with h5py.File(path, "r") as h5:
        for name in ("success", "fsm_done", "fps", "joint_names"):
            if name not in h5.attrs:
                raise ValueError(f"{path.name}: missing HDF5 attr '{name}'")
        success = int(h5.attrs["success"])
        if success not in (-1, 0, 1):
            raise ValueError(f"{path.name}: invalid success={success}")
        fps = int(h5.attrs["fps"])
        if fps != FPS:
            raise ValueError(f"{path.name}: expected {FPS}Hz but got {fps}Hz")
        names = json.loads(decode_attr(h5.attrs["joint_names"]))
        if len(names) != 12 or len(set(names)) != 12:
            raise ValueError(f"{path.name}: expected 12 distinct joint names, got {names}")
        if "action" not in h5:
            raise ValueError(f"{path.name}: no action dataset")
        n = int(h5["action"].shape[0])
        if n <= 1:
            raise ValueError(f"{path.name}: too few frames ({n})")
        fmt = decode_attr(h5.attrs.get("action_format", ""))
        if fmt != ",".join(ACTION_NAMES):
            raise ValueError(f"{path.name}: unexpected action_format={fmt!r}")
        return Episode(
            path=path,
            source_id=episode_id(path),
            success=success,
            fsm_done=bool(h5.attrs["fsm_done"]),
            frames=n,
            fps=fps,
            joint_names=tuple(names),
            reason=decode_attr(h5.attrs.get("termination_reason", "")),
        )


def validate_hdf5_episode(h5: h5py.File, ep: Episode) -> None:
    """在 LeRobot 创建前，校验新采集器的各模态时间与形状。"""
    n = ep.frames
    shapes = {
        "action": (n, 8),
        "observations/qpos": (n, 12),
        "observations/qvel": (n, 12),
        "observations/ee_pose": (n, 7),
        "object_position": (n, 3),
        "timestamp": (n,),
        "fsm_state": (n,),
        "observations/images/wrist": (n, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        "observations/images/front": (n, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        "observations/depth/wrist": (n, IMAGE_HEIGHT, IMAGE_WIDTH),
        "observations/depth/front": (n, IMAGE_HEIGHT, IMAGE_WIDTH),
    }
    for key, shape in shapes.items():
        if key not in h5:
            raise KeyError(f"{ep.path.name}: missing dataset '{key}'")
        if h5[key].shape != shape:
            raise ValueError(
                f"{ep.path.name}: {key} shape={h5[key].shape}, expected={shape}"
            )
    for key in ("observations/images/wrist", "observations/images/front"):
        if h5[key].dtype != np.dtype("uint8"):
            raise ValueError(f"{ep.path.name}: {key} should be uint8")
    for key in DEPTH_KEYS.values():
        if h5[key].dtype != np.dtype("float32"):
            raise ValueError(f"{ep.path.name}: {key} should be float32 metres")
    for key in ("action", "observations/qpos", "observations/qvel",
                "observations/ee_pose", "object_position"):
        if not np.isfinite(h5[key][:]).all():
            raise ValueError(f"{ep.path.name}: non-finite values in {key}")

    ts = np.asarray(h5["timestamp"][:], dtype=np.float64)
    if not np.isfinite(ts).all() or not np.allclose(
        np.diff(ts), 1.0 / FPS, rtol=1e-4, atol=1e-5
    ):
        raise ValueError(f"{ep.path.name}: timestamps are not spaced at 0.04s")

    for key in ("action", "observations/ee_pose"):
        q = h5[key][:, 3:7]
        norm = np.linalg.norm(q, axis=1)
        if not np.allclose(norm, 1.0, atol=0.03):
            raise ValueError(f"{ep.path.name}: non-unit WXYZ quaternion in {key}")
    # 深度允许 Inf/NaN，原始 HDF5 原样保留。若转换为 LeRobot depth video，
    # 那时才在副本中将无效值标记为 10m；不会写回 HDF5。


def features(joint_names, include_depth=False):
    result = {
        "observation.state": {
            "dtype": "float32",
            "shape": (34,),
            "names": build_state_names(joint_names),
        },
        "action": {"dtype": "float32", "shape": (8,), "names": ACTION_NAMES},
        "observation.images.wrist": {
            "dtype": "video", "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.front": {
            "dtype": "video", "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
    }
    if include_depth:
        for name in ("wrist", "front"):
            result[f"observation.depth.{name}"] = {
                "dtype": "video",
                "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 1),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": True, "depth_unit": "m"},
            }
    return result


def convert_depth_image(depth: np.ndarray):
    # LeRobot 0.7.0 depth video is quantized. Keep original float32 HDF5 untouched.
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=10.0, posinf=10.0, neginf=0.01)
    depth = np.clip(depth, 0.01, 10.0)
    return np.ascontiguousarray(depth[..., None], dtype=np.float32)


def convert_split(episodes: list[Episode], split: str, root: Path, repo_prefix: str,
                  include_depth: bool, joint_names: list[str]) -> int:
    # 延迟导入：仅 --scan-only 时不需要安装 LeRobot。
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=f"{repo_prefix}_{split}",
        root=root / split,
        fps=FPS,
        robot_type="lebai_lm3_lmg90",
        features=features(joint_names, include_depth),
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=2,
        batch_encoding_size=1,
    )
    try:
        for i, ep in enumerate(episodes):
            with h5py.File(ep.path, "r") as h5:
                validate_hdf5_episode(h5, ep)
                for frame_index in range(ep.frames):
                    state = build_state(
                        h5["observations/qpos"][frame_index],
                        h5["observations/qvel"][frame_index],
                        h5["observations/ee_pose"][frame_index],
                        h5["object_position"][frame_index],
                    )
                    frame = {
                        "observation.state": state,
                        "action": np.asarray(h5["action"][frame_index], dtype=np.float32),
                        "observation.images.wrist": np.ascontiguousarray(
                            h5[RGB_KEYS["wrist"]][frame_index]
                        ),
                        "observation.images.front": np.ascontiguousarray(
                            h5[RGB_KEYS["front"]][frame_index]
                        ),
                        "task": TASK,
                    }
                    if include_depth:
                        for name in ("wrist", "front"):
                            frame[f"observation.depth.{name}"] = convert_depth_image(
                                h5[DEPTH_KEYS[name]][frame_index]
                            )
                    dataset.add_frame(frame)
                dataset.save_episode(parallel_encoding=False)
            print(f"[{split}] {i+1}/{len(episodes)} {ep.path.name} "
                  f"frames={ep.frames}", flush=True)
    except Exception:
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer(delete_images=True)
        raise
    finally:
        dataset.finalize()
    if dataset.meta.total_episodes != len(episodes):
        raise RuntimeError(f"{split}: episode count mismatch")
    return int(dataset.meta.total_frames)


def make_manifest(args, statuses, selected, train, val, train_frames, val_frames):
    return {
        "converter": "convert_hdf5_to_lerobot_25hz.py",
        "fps": FPS,
        "seed": args.seed,
        "source_dir": str(args.input_dir.resolve()),
        "repo_prefix": args.repo_prefix,
        "state_semantics": "qpos12+qvel12+ee_pose7+object_position_minus_ee_xyz3",
        "action_semantics": "absolute XYZ+WXYZ+gripper, 8D",
        "depth_in_lerobot": bool(args.include_depth),
        "depth_source_hdf5_keys": DEPTH_KEYS,
        "rgb_source_hdf5_keys": RGB_KEYS,
        "depth_info": (
            "Original full-precision float32 metric depth remains in source HDF5. "
            "If include-depth=true, LeRobot 12-bit quantized depth videos are created "
            "with non-finite pixels mapped to 10m in the converted copy."
        ),
        "selection_rule": "success == 1 AND fsm_done == True",
        "scanned_episodes": [
            {"source_episode_id": ep.source_id, "source_file": ep.path.name,
             "success": ep.success, "fsm_done": ep.fsm_done,
             "frames": ep.frames, "termination_reason": ep.reason}
            for ep in statuses
        ],
        "selected_successful_source_ids": [ep.source_id for ep in selected],
        "train_source_episode_ids": [ep.source_id for ep in train],
        "val_source_episode_ids": [ep.source_id for ep in val],
        "train_dataset_episode_to_source": [ep.source_id for ep in train],
        "val_dataset_episode_to_source": [ep.source_id for ep in val],
        "train_frames": train_frames,
        "val_frames": val_frames,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/raw_25hz"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("datasets/lerobot/lebai_lm3_pick_place_25hz"))
    parser.add_argument("--repo-prefix", type=str,
                        default="local/lebai_lm3_pick_place_25hz")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-episodes", type=int, default=160)
    parser.add_argument("--val-episodes", type=int, default=40)
    parser.add_argument("--allow-fewer-successes", action="store_true",
                        help="Use all eligible successes in an 80/20 episode split if fewer than target")
    parser.add_argument("--include-depth", action="store_true",
                        help="Encode metric float32 depth as LeRobot native depth videos (requires codec)")
    parser.add_argument("--scan-only", action="store_true",
                        help="Validate metadata, print status, and exit without converting")
    args = parser.parse_args()

    if args.train_episodes < 1 or args.val_episodes < 1:
        parser.error("--train-episodes and --val-episodes must both be positive")
    files = sorted(args.input_dir.glob("episode_*.hdf5"))
    if not files:
        raise RuntimeError(f"No HDF5 files found in {args.input_dir}")
    statuses = [scan_metadata(path) for path in files]
    ids = [ep.source_id for ep in statuses]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate source episode IDs detected")
    joint_names = list(statuses[0].joint_names)
    if any(ep.joint_names != statuses[0].joint_names for ep in statuses):
        raise ValueError("Joint order differs across HDF5 files")

    eligible = [ep for ep in statuses if ep.success == 1 and ep.fsm_done]
    counts = {
        "success_completed": len(eligible),
        "failed": sum(ep.success == 0 for ep in statuses),
        "unknown": sum(ep.success == -1 for ep in statuses),
        "fsm_incomplete": sum(not ep.fsm_done for ep in statuses),
    }
    print(f"[SCAN] Source={args.input_dir}; {len(statuses)} HDF5 files; "
          f"status={json.dumps(counts)}; fps={FPS}")
    for ep in eligible:
        with h5py.File(ep.path, "r") as h5:
            validate_hdf5_episode(h5, ep)
    print(f"[SCAN] Validated {len(eligible)} success episodes (RGB + depth + state + timestamps).")
    if args.scan_only:
        return

    wanted = args.train_episodes + args.val_episodes
    if len(eligible) < wanted and not args.allow_fewer_successes:
        raise RuntimeError(
            f"Only {len(eligible)} successful, FSM-completed demos; need {wanted} "
            f"({args.train_episodes} train + {args.val_episodes} val). "
            "The 25Hz collector counts FSM-completed episodes, NOT verified successes. "
            "Collect additional successes or use --allow-fewer-successes."
        )
    if len(eligible) < 2:
        raise RuntimeError("At least 2 successful episodes are required for train/val split")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing dataset directory: {args.output_dir}")

    rng = np.random.default_rng(args.seed)
    eligible = [eligible[int(i)] for i in rng.permutation(len(eligible))]
    if len(eligible) >= wanted:
        selected = eligible[:wanted]
        n_train = args.train_episodes
    else:
        selected = eligible
        ratio = args.val_episodes / wanted
        n_val = max(1, round(len(selected) * ratio))
        n_train = len(selected) - n_val
    train, val = selected[:n_train], selected[n_train:]
    if set(ep.source_id for ep in train) & set(ep.source_id for ep in val):
        raise RuntimeError("Train/validation source episode overlap detected")
    print(f"[SPLIT] {len(train)} train / {len(val)} val, "
          f"episodes success-only, seed={args.seed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    train_frames = convert_split(train, "train", args.output_dir, args.repo_prefix,
                                 args.include_depth, joint_names)
    val_frames = convert_split(val, "val", args.output_dir, args.repo_prefix,
                               args.include_depth, joint_names)
    manifest = make_manifest(args, statuses, selected, train, val, train_frames, val_frames)
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"[DONE] {len(train)} train / {len(val)} val, "
          f"{train_frames + val_frames} total frames, fps={FPS}\n"
          f"       Manifest: {manifest_path}")
    print("[NOTE] Standard ACT RGB model ignores raw depth. "
          "Use --include-depth only for a separate RGB-D experiment with adapted model inputs.")


if __name__ == "__main__":
    main()

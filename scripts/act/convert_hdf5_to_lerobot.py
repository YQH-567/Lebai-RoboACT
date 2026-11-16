
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


ACTION_NAMES = [
    "x", "y", "z",
    "qw", "qx", "qy", "qz",
    "gripper",
]

IMAGE_KEYS = {
    "wrist": "observation.images.wrist",
    "front": "observation.images.front",
}

TASK = "Pick up the cube and place it into the tray."

def build_state_names(joint_names):
    return (
        [f"qpos_{name}" for name in joint_names]
        + [f"qvel_{name}" for name in joint_names]
        + [
            "ee_x", "ee_y", "ee_z",
            "ee_qw", "ee_qx", "ee_qy", "ee_qz"
        ]
    )

def decode_attr(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def load_joint_names(value):
    names = json.loads(decode_attr(value))
    if len(names) != 12:
        raise ValueError(f"Expected 12 joint names, got {len(names)}")
    return names


def validate_episode(f, path, expected_fps=None):

    action = f["action"][:]
    qpos = f["observations/qpos"][:]
    qvel = f["observations/qvel"][:]
    ee_pose = f["observations/ee_pose"][:]
    timestamp = f["timestamp"][:]    

    wrist = f["observations/images/wrist"]
    front = f["observations/images/front"]#这里是你的输入

    n = len(action)

    if action.shape != (n, 8):
        raise ValueError(f"{path.name}: invalid action shape")

    if qpos.shape != (n, 12):
        raise ValueError(f"{path.name}: invalid qpos shape")

    if wrist.shape != (n, 180, 320, 3):
        raise ValueError(f"{path.name}: invalid wrist image shape")

    if front.shape != (n, 180, 320, 3):
        raise ValueError(f"{path.name}: invalid front image shape")

    if wrist.dtype != np.uint8 or front.dtype != np.uint8:
        raise ValueError(f"{path.name}: images must be uint8")

    if not np.isfinite(action).all():
        raise ValueError(f"{path.name}: invalid action values")

    if not np.isfinite(qpos).all():
        raise ValueError(f"{path.name}: invalid qpos values")

    if not np.isfinite(timestamp).all():
        raise ValueError(f"{path.name}: invalid timestamps")

    fps = int(f.attrs["fps"])

    if expected_fps is not None and fps != expected_fps:
        raise ValueError(f"{path.name}: inconsistent FPS")

    if not bool(f.attrs["fsm_done"]):
        raise ValueError(f"{path.name}: FSM did not finish")

    action_format = decode_attr(f.attrs["action_format"])

    if action_format != "x,y,z,qw,qx,qy,qz,gripper":
        raise ValueError(f"{path.name}: unexpected action format")

    joint_names = load_joint_names(f.attrs["joint_names"])

    # Quaternion unit norm check: HDF5 uses WXYZ.
    quat = action[:, 3:7]
    quat_norm = np.linalg.norm(quat, axis=1)

    if not np.allclose(quat_norm, 1.0, atol=0.03):
        raise ValueError(
            f"{path.name}: quaternion is not normalized"
        )

    # Check quaternion sign discontinuities.
    dots = np.sum(quat[1:] * quat[:-1], axis=1)
    sign_flips = int(np.sum(dots < -1e-6))

    if sign_flips:
        print(
            f"[WARNING] {path.name}: "
            f"{sign_flips} quaternion sign flips"
        )

    # Verify the actual recording interval.
    intervals = np.diff(timestamp)

    if len(intervals) and not np.allclose(
        intervals, 1.0 / fps, atol=0.003, rtol=0.01
    ):
        print(
            f"[WARNING] {path.name}: "
            f"timestamp interval differs from {1.0 / fps:.3f}s"
        )

    if qvel.shape != (n, 12):
        raise ValueError(
            f"{path.name}: invalid qvel shape {qvel.shape}"
        )

    if ee_pose.shape != (n, 7):
        raise ValueError(
            f"{path.name}: invalid ee_pose shape {ee_pose.shape}"
        )

    if not np.isfinite(qvel).all():
        raise ValueError(f"{path.name}: invalid qvel values")

    if not np.isfinite(ee_pose).all():
        raise ValueError(f"{path.name}: invalid ee_pose values")

    # 检查 EE 四元数是否归一化
    ee_quat_norm = np.linalg.norm(ee_pose[:, 3:7], axis=1)

    if not np.allclose(ee_quat_norm, 1.0, atol=0.03):
        raise ValueError(
            f"{path.name}: EE quaternion is not normalized"
        )

    #return n, fps, joint_names, action, qpos, wrist, front   
    return (
        n, fps, joint_names,
        action, qpos, qvel, ee_pose,
        wrist, front
    )#修改了输入记得返回值也要修改，同时这些数据检查最好做一下

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("datasets/raw"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/lerobot/lebai_lm3_pick_place_v1"),
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/lebai_lm3_pick_place_v1",
    )

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Only convert the first N episodes for testing.",
    )

    args = parser.parse_args()

    files = sorted(
        args.input_dir.glob("episode_*.hdf5")
    )

    if not files:
        raise RuntimeError(
            f"No HDF5 episodes found in {args.input_dir}"
        )

    if args.max_episodes is not None:
        files = files[:args.max_episodes]

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}"
        )

    print(f"[INFO] Episodes: {len(files)}")

    # Read metadata from the first episode.
    with h5py.File(files[0], "r") as f:
        _, fps, joint_names, *_ = validate_episode(
            f, files[0]
        )

    features = {

        # "observation.state": {
        #     "dtype": "float32",
        #     "shape": (12,),
        #     "names": joint_names,
        # },
        #还有修改这里的
        "observation.state": {
            "dtype": "float32",
            "shape": (31,),
            "names": build_state_names(joint_names),
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": ACTION_NAMES,
        },

        "observation.images.wrist": {
            "dtype": "video",
            "shape": (180, 320, 3),
            "names": ["height", "width", "channels"],
        },

        "observation.images.front": {
            "dtype": "video",
            "shape": (180, 320, 3),
            "names": ["height", "width", "channels"],
        },
    }

    args.output_dir.parent.mkdir(
        parents=True, exist_ok=True
    )

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_dir,
        fps=fps,
        robot_type="lebai_lm3_lmg90",
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=2,
        batch_encoding_size=1,
    )

    total_frames = 0

    try:

        for ep_idx, path in enumerate(files):

            with h5py.File(path, "r") as f:

                # n, ep_fps, names, action, qpos, wrist, front = (
                #     validate_episode(
                #         f, path, expected_fps=fps
                #     )
                # )
                #拼接数据也需要更改：
                n, ep_fps, names,action, qpos, qvel, ee_pose,wrist, front = validate_episode(f, path, expected_fps=fps)

                if names != joint_names:
                    raise ValueError(
                        f"{path.name}: joint order changed"
                    )

                for i in range(n):

                    frame = {

                        # "observation.state":
                        #     np.asarray(
                        #         qpos[i], dtype=np.float32
                        #     ),
                        # 这里也需要更改
                        "observation.state": np.concatenate([
                            qpos[i],
                            qvel[i],
                            ee_pose[i],
                        ]).astype(np.float32),

                        "action":
                            np.asarray(
                                action[i], dtype=np.float32
                            ),

                        "observation.images.wrist":
                            np.ascontiguousarray(wrist[i]),

                        "observation.images.front":
                            np.ascontiguousarray(front[i]),

                        "task": TASK,
                    }

                    dataset.add_frame(frame)

                # One HDF5 episode = one LeRobot episode.
                dataset.save_episode(
                    parallel_encoding=False
                )

                total_frames += n

            print(
                f"[CONVERTED] "
                f"{ep_idx + 1}/{len(files)} "
                f"{path.name} frames={n}"
            )

    except Exception:

        if dataset.has_pending_frames():
            dataset.clear_episode_buffer(
                delete_images=True
            )

        raise

    finally:
        dataset.finalize()

    print("\n========== CONVERSION COMPLETE ==========")
    print("Episodes:", dataset.meta.total_episodes)
    print("Frames:", dataset.meta.total_frames)
    print("FPS:", fps)
    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()

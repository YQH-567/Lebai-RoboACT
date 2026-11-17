"""HDF5 demonstration recorder for synchronized 25Hz RGB-D ACT data.

RGB datasets:
    observations/images/wrist  uint8, HWC, RGB
    observations/images/front  uint8, HWC, RGB

Depth datasets:
    observations/depth/wrist   float32, HW, distance_to_image_plane in meters
    observations/depth/front   float32, HW, distance_to_image_plane in meters

State/action datasets retain the original absolute-IK format.
"""

from pathlib import Path
import json

import h5py
import numpy as np


class DemoRecorder:
    def __init__(self, save_dir, fps=25, compression_level=2):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.compression_level = int(compression_level)

        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if not 0 <= self.compression_level <= 9:
            raise ValueError("compression_level must be between 0 and 9.")

        existing_ids = [
            int(p.stem.split("_")[-1])
            for p in self.save_dir.glob("episode_*.hdf5")
            if p.stem.split("_")[-1].isdigit()
        ]
        self.episode_id = max(existing_ids, default=-1) + 1
        self.joint_names = None
        self.reset()

    @property
    def has_data(self):
        return bool(self.buffer["action"])

    def reset(self):
        self.buffer = {
            "observations/images/wrist": [],
            "observations/images/front": [],
            "observations/depth/wrist": [],
            "observations/depth/front": [],
            "observations/qpos": [],
            "observations/qvel": [],
            "observations/ee_pose": [],
            "action": [],
            "timestamp": [],
            "fsm_state": [],
            "object_position": [],
            "object_velocity": [],
            "object_speed": [],
            "inside_tray": [],
            "success_condition": [],
            "success_stable_count": [],
        }

    @staticmethod
    def to_numpy(value, dtype=None):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return np.array(array, copy=True)

    @classmethod
    def _read_rgb(cls, camera, camera_name):
        output = camera.data.output
        if "rgb" not in output:
            raise KeyError(
                f"{camera_name} camera has no 'rgb' output. "
                f"Available keys: {list(output.keys())}"
            )
        rgb = cls.to_numpy(output["rgb"][0], dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(
                f"{camera_name} RGB must be HxWx3, got {rgb.shape}"
            )
        return rgb

    @classmethod
    def _read_depth(cls, camera, camera_name):
        output = camera.data.output
        key = "distance_to_image_plane"
        if key not in output:
            raise KeyError(
                f"{camera_name} camera has no '{key}' output. "
                f"Available keys: {list(output.keys())}"
            )

        depth = cls.to_numpy(output[key][0], dtype=np.float32)

        # Isaac Lab versions may expose depth as HxW or HxWx1.
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]

        if depth.ndim != 2:
            raise ValueError(
                f"{camera_name} depth must be HxW or HxWx1, got {depth.shape}"
            )
        return depth

    def add(
        self,
        scene,
        action,
        ee_pose,
        fsm_state,
        timestamp,
        env_origins,
        object_position,
        object_velocity,
        object_speed,
        inside_tray,
        success_condition,
        success_stable_count,
    ):
        robot = scene["robot"]
        wrist = scene["wrist_camera"]
        front = scene["front_camera"]

        if self.joint_names is None:
            self.joint_names = list(robot.joint_names)

        wrist_rgb = self._read_rgb(wrist, "wrist")
        front_rgb = self._read_rgb(front, "front")
        wrist_depth = self._read_depth(wrist, "wrist")
        front_depth = self._read_depth(front, "front")

        if wrist_rgb.shape[:2] != wrist_depth.shape:
            raise ValueError(
                f"Wrist RGB/depth size mismatch: {wrist_rgb.shape} vs {wrist_depth.shape}"
            )
        if front_rgb.shape[:2] != front_depth.shape:
            raise ValueError(
                f"Front RGB/depth size mismatch: {front_rgb.shape} vs {front_depth.shape}"
            )

        self.buffer["observations/images/wrist"].append(wrist_rgb)
        self.buffer["observations/images/front"].append(front_rgb)
        self.buffer["observations/depth/wrist"].append(wrist_depth)
        self.buffer["observations/depth/front"].append(front_depth)

        self.buffer["observations/qpos"].append(
            self.to_numpy(robot.data.joint_pos[0], dtype=np.float32)
        )
        self.buffer["observations/qvel"].append(
            self.to_numpy(robot.data.joint_vel[0], dtype=np.float32)
        )
        self.buffer["observations/ee_pose"].append(
            self.to_numpy(ee_pose[0], dtype=np.float32)
        )
        self.buffer["action"].append(
            self.to_numpy(action[0], dtype=np.float32)
        )
        self.buffer["timestamp"].append(float(timestamp))
        self.buffer["fsm_state"].append(int(fsm_state))
        self.buffer["object_position"].append(
            self.to_numpy(object_position[0], dtype=np.float32)
        )
        self.buffer["object_velocity"].append(
            self.to_numpy(object_velocity, dtype=np.float32)
        )
        self.buffer["object_speed"].append(float(object_speed))
        self.buffer["inside_tray"].append(bool(inside_tray))
        self.buffer["success_condition"].append(bool(success_condition))
        self.buffer["success_stable_count"].append(int(success_stable_count))

        lengths = {key: len(values) for key, values in self.buffer.items()}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"Misaligned HDF5 fields after append: {lengths}")

    def save(
        self,
        success=None,
        fsm_done=False,
        termination_reason="unknown",
        success_criterion=None,
    ):
        if not self.has_data:
            return None

        lengths = {key: len(values) for key, values in self.buffer.items()}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"Cannot save misaligned episode: {lengths}")

        timestamps = np.asarray(self.buffer["timestamp"], dtype=np.float64)
        expected_dt = 1.0 / self.fps
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            if not np.allclose(intervals, expected_dt, rtol=1e-4, atol=1e-6):
                raise ValueError(
                    "Recorded timestamps are not uniformly sampled at the "
                    f"declared {self.fps}Hz. Expected {expected_dt:.6f}s, "
                    f"observed range [{intervals.min():.6f}, {intervals.max():.6f}]s."
                )

        path = self.save_dir / f"episode_{self.episode_id:06d}.hdf5"

        with h5py.File(path, "w") as file:
            for key, values in self.buffer.items():
                data = np.stack(values)
                file.create_dataset(
                    key,
                    data=data,
                    compression="gzip",
                    compression_opts=self.compression_level,
                    shuffle=True,
                )

            file.attrs["fps"] = self.fps
            file.attrs["camera_fps"] = self.fps
            file.attrs["state_action_fps"] = self.fps
            file.attrs["num_frames"] = len(self.buffer["action"])
            file.attrs["episode_duration_s"] = (
                float(timestamps[-1]) if len(timestamps) else 0.0
            )
            file.attrs["fsm_done"] = bool(fsm_done)

            # success: 1 = geometrically verified placement;
            #          0 = completed/terminated but not successful;
            #         -1 = outcome unknown (e.g. interrupted recording).
            file.attrs["success"] = -1 if success is None else int(success)
            file.attrs["success_semantics"] = (
                "1=object settled inside approximate tray region with gripper open; "
                "0=not verified successful; -1=unknown/interrupted"
            )
            file.attrs["success_criterion"] = json.dumps(
                success_criterion or {}, ensure_ascii=False
            )
            file.attrs["termination_reason"] = str(termination_reason)

            file.attrs["joint_names"] = json.dumps(self.joint_names or [])
            file.attrs["action_format"] = "x,y,z,qw,qx,qy,qz,gripper"
            file.attrs["action_semantics"] = (
                "absolute end-effector target pose in environment-local coordinates; "
                "quaternion WXYZ; gripper +1=open, -1=close"
            )
            file.attrs["rgb_format"] = "uint8 HWC RGB"
            file.attrs["depth_dataset_paths"] = (
                "observations/depth/wrist;observations/depth/front"
            )
            file.attrs["depth_encoding"] = "distance_to_image_plane"
            file.attrs["depth_dtype"] = "float32"
            file.attrs["depth_unit"] = "meter"
            file.attrs["depth_layout"] = "HW"
            file.attrs["timestamp_unit"] = "second"
            file.attrs["timestamp_semantics"] = (
                "elapsed episode time at the observation/action sample"
            )

        print(
            f"[DATASET] Saved {path.name}, "
            f"frames={len(self.buffer['action'])}, "
            f"fps={self.fps}, success={-1 if success is None else int(success)}, "
            f"fsm_done={bool(fsm_done)}"
        )

        self.episode_id += 1
        self.reset()
        return path

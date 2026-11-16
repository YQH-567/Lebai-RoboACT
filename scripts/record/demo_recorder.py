
from pathlib import Path
import json

import h5py
import numpy as np


class DemoRecorder:
    def __init__(self, save_dir, fps=10):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        existing_ids = [
            int(p.stem.split("_")[-1])
            for p in self.save_dir.glob("episode_*.hdf5")
            if p.stem.split("_")[-1].isdigit()
        ]

        self.episode_id = max(existing_ids, default=-1) + 1
        self.joint_names = None

        self.reset()

    def reset(self):
        self.buffer = {
            "observations/images/wrist": [],
            "observations/images/front": [],
            "observations/qpos": [],
            "observations/qvel": [],
            "observations/ee_pose": [],
            "action": [],
            "timestamp": [],
            "fsm_state": [],
            "object_position": [],
        }

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy().copy()

    def add(self, scene, action, ee_pose,
            fsm_state, timestamp, env_origins):

        robot = scene["robot"]
        if self.joint_names is None:
            self.joint_names = list(robot.joint_names)
        wrist = scene["wrist_camera"]
        front = scene["front_camera"]
        obj = scene["object"]

        self.buffer["observations/images/wrist"].append(
            self.to_numpy(wrist.data.output["rgb"][0])
        )
        self.buffer["observations/images/front"].append(
            self.to_numpy(front.data.output["rgb"][0])
        )

        self.buffer["observations/qpos"].append(
            self.to_numpy(robot.data.joint_pos[0])
        )
        self.buffer["observations/qvel"].append(
            self.to_numpy(robot.data.joint_vel[0])
        )
        self.buffer["observations/ee_pose"].append(
            self.to_numpy(ee_pose[0])
        )

        self.buffer["action"].append(
            self.to_numpy(action[0])
        )
        self.buffer["timestamp"].append(float(timestamp))
        self.buffer["fsm_state"].append(int(fsm_state))

        self.buffer["object_position"].append(
            self.to_numpy(
                obj.data.root_pos_w[0] - env_origins[0]
            )
        )

    def save(self,success=None, fsm_done=False,termination_reason="unknown",):
        if not self.buffer["action"]:
            return None

        path = self.save_dir / (
            f"episode_{self.episode_id:06d}.hdf5"
        )

        with h5py.File(path, "w") as file:
            for key, values in self.buffer.items():
                data = np.stack(values)

                file.create_dataset(
                    key,
                    data=data,
                    compression="gzip",
                    compression_opts=2,
                )

            file.attrs["fps"] = self.fps
            file.attrs["num_frames"] = len(
                self.buffer["action"]
            )
            file.attrs["fsm_done"] = bool(fsm_done)

            # -1: 尚未人工或自动验证成功
            file.attrs["success"] = (
                -1 if success is None else int(success)
            )

            file.attrs["termination_reason"] = termination_reason

            file.attrs["joint_names"] = json.dumps(
                self.joint_names or []
            )

            file.attrs["action_format"] = (
                "x,y,z,qw,qx,qy,qz,gripper"
            )

            file.attrs["joint_names"] = json.dumps(
                self.joint_names or []
            )

        print(
            f"[DATASET] Saved {path.name}, "
            f"frames={len(self.buffer['action'])}, "
            f"fsm_done={fsm_done}"
        )

        self.episode_id += 1
        self.reset()
        return path

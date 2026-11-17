
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lebai LM3 + ACT simulation evaluation (v7, 34D state).

Compared with v2, append object_position - actual_ee_position to the state.
Both positions are measured in environment-local meters. Action, cameras,
absolute IK, and safety limits retain the v2 behavior.

相对 eval_act_isaaclab.py（v1）的主要修改：
1. 收紧 X 轴工作空间上限：默认从 0.55 m 改为 0.52 m，
   依据当前示教数据 X 最大值约 0.5023 m，额外保留约 1.8 cm 余量；
   可通过 --workspace-x-max 调整。Y/Z 工作空间边界不变。
2. 保留原有 0.5 m/s 末端笛卡尔目标步长限制，不擅自取消速度保护。
3. 将物块速度拆成两项分别记录：
   - object_speed_displacement：相邻两个 Policy Step 的位置差分速度；
   - object_speed_physx：PhysX 刚体瞬时线速度。
   成功稳定性判断使用位置差分速度，不再混用两种速度。
4. 每个 Policy Step 额外记录实际 EE 位姿、EE 坐标系原点到物块根位置（中心参考点）的距离、
   EE-原始 ACT 目标距离、EE-限幅后执行目标距离，便于定位误差来源。
5. 分别记录工作空间限幅、速度限幅和总限幅标志；保留原始/执行动作。
6. 默认输出目录为 outputs/eval/act_lebai_lm3_v7。
7. 动作日志统一先转为 NumPy 数组，兼容 worker 返回 list 或 ndarray。

Isaac Lab process:
    env_isaaclab (Python 3.11)
    - launches Isaac Sim
    - reads wrist/front RGB and 34D state
    - executes ACT's absolute Cartesian actions through env.step()

Inference worker:
    env_act (Python 3.12)
    - loads LeRobot ACT checkpoint and saved processors
    - maintains ACT action queue
    - returns one unnormalized 8D action per request

No real robot SDK or network connection is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import socket
import struct
import subprocess
import sys
import time
import traceback

from pathlib import Path

from act_v7_state import build_state, build_state_names


# ============================================================
# 共享配置
# ============================================================

ENV_ID = "Isaac-Lift-Cube-Lebai-IK-Abs-v0"

JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper_l_joint1",
    "gripper_l_joint2",
    "gripper_r_joint1",
    "gripper_r_joint2",
    "gripper_l_joint_finger",
    "gripper_r_joint_finger",
]

STATE_NAMES = build_state_names(JOINT_NAMES)

ACTION_NAMES = [
    "x", "y", "z",
    "qw", "qx", "qy", "qz",
    "gripper",
]

TASK = "Pick up the cube and place it into the tray."

DATASET_FEATURES = {
    # "observation.state": {
    #     "dtype": "float32",
    #     "shape": (12,),
    #     "names": JOINT_NAMES,
    # },
    "observation.state": {
        "dtype": "float32",
        "shape": (34,),
        "names": STATE_NAMES,
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

    "action": {
        "dtype": "float32",
        "shape": (8,),
        "names": ACTION_NAMES,
    },
}


# ============================================================
# TCP 协议：带长度前缀的 pickle，仅限本地主机
# ============================================================

def send_packet(sock, obj):
    payload = pickle.dumps(
        obj,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    sock.sendall(struct.pack("!Q", len(payload)))
    sock.sendall(payload)


def recv_exact(sock, size):
    chunks = []
    remaining = size

    while remaining > 0:
        chunk = sock.recv(remaining)

        if not chunk:
            raise ConnectionError("Inference worker disconnected")

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def recv_packet(sock):
    header = recv_exact(sock, 8)
    size = struct.unpack("!Q", header)[0]

    if size > 20 * 1024 * 1024:
        raise ValueError("Invalid packet size")

    payload = recv_exact(sock, size)

    return pickle.loads(payload)


# ============================================================
# ACT推理工作者：仅在env_act环境中运行
# ============================================================

def run_inference_worker(args):

    import numpy as np
    import torch

    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import build_inference_frame

    checkpoint = Path(args.checkpoint).expanduser().resolve()

    if not (checkpoint / "model.safetensors").exists():
        raise FileNotFoundError(
            f"ACT weights not found: {checkpoint}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("[ACT WORKER] Loading checkpoint:", checkpoint)
    print("[ACT WORKER] Device:", device)

    # Load ACT weights.
    model = ACTPolicy.from_pretrained(
        pretrained_name_or_path=str(checkpoint),
        local_files_only=True,
    )

    model = model.to(device)
    model.eval()

    # # 评估阶段启用每步重新规划
    # # 不修改训练好的模型权重

    # # 必须重置动作队列，使队列长度同步变成1
    # model.reset()

    # print("[ACT WORKER] Runtime action steps:",
    #     model.config.n_action_steps)
    # Experiment: execute 20 actions per ACT inference
    model.config.n_action_steps = 30

    # Rebuild internal action queue using the new configuration
    model.reset()

    assert model._action_queue.maxlen == 30

    print("[ACT WORKER] Chunk size:",
        model.config.chunk_size)

    print("[ACT WORKER] Action steps:",
        model.config.n_action_steps)

    print("[ACT WORKER] Actual queue maxlen:",
        model._action_queue.maxlen)

    # Load the exact saved normalization pipelines.
    # Do NOT calculate new statistics.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {
                "device": str(device),
            }
        },
    )

    print("[ACT WORKER] Chunk size:",
          model.config.chunk_size)

    print("[ACT WORKER] Action steps:",
          model.config.n_action_steps)

    print("[ACT WORKER] Preprocessor and postprocessor loaded.")

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    )

    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1,
    )

    server.bind(("127.0.0.1", args.port))
    server.listen(1)

    print(
        f"[ACT WORKER] Listening on 127.0.0.1:{args.port}",
        flush=True,
    )

    try:

        conn, address = server.accept()

        print("[ACT WORKER] Connected:", address)

        with conn:

            conn.settimeout(120)

            send_packet(
                conn,
                {
                    "status": "ready",
                    "device": str(device),
                    "chunk_size": model.config.chunk_size,
                    "n_action_steps": model.config.n_action_steps,
                },
            )

            while True:

                request = recv_packet(conn)

                command = request["cmd"]

                if command == "close":

                    send_packet(conn, {"status": "closed"})
                    break

                if command == "reset":

                    # Clear ACT's internal action queue.
                    model.reset()

                    send_packet(
                        conn,
                        {"status": "reset_ok"},
                    )

                    continue

                if command != "infer":
                    raise ValueError(
                        f"Unknown worker command: {command}"
                    )

                try:

                    raw = request["observation"]

                    # qpos = np.asarray(
                    #     raw["qpos"],
                    #     dtype=np.float32,
                    # )
                    state = np.asarray(
                        raw["state"],
                        dtype=np.float32,
                    )

                    if state.shape != (34,):
                        raise ValueError(
                            f"Invalid state shape: {state.shape}"
                        )

                    wrist = np.asarray(
                        raw["wrist"],
                        dtype=np.uint8,
                    )

                    front = np.asarray(
                        raw["front"],
                        dtype=np.uint8,
                    )

                    # if qpos.shape != (12,):
                    #     raise ValueError(
                    #         f"Invalid qpos shape: {qpos.shape}"
                    #     )

                    if wrist.shape != (180, 320, 3):
                        raise ValueError(
                            f"Invalid wrist shape: {wrist.shape}"
                        )

                    if front.shape != (180, 320, 3):
                        raise ValueError(
                            f"Invalid front shape: {front.shape}"
                        )

                    # The helper constructs the exact LeRobot
                    # observation feature dictionary:
                    #
                    # HWC uint8 -> CHW float32 [0,1]
                    # state -> (1,34)
                    #
                    # The saved preprocessor then applies
                    # the checkpoint's normalization.

                    values = {
                        name: float(state[i])
                        for i, name in enumerate(STATE_NAMES)
                    }

                    values["wrist"] = wrist
                    values["front"] = front

                    observation_frame = build_inference_frame(
                        observation=values,
                        device=device,
                        ds_features=DATASET_FEATURES,
                        task=TASK,
                    )

                    processed = preprocessor(
                        observation_frame
                    )

                    # select_action manages ACT's internal
                    # 10-step action queue.
                    with torch.inference_mode():

                        action = model.select_action(
                            processed
                        )

                        # Restore original Cartesian action scale.
                        action = postprocessor(action)

                    if isinstance(action, dict):
                        action = action["action"]

                    action = (
                        action
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    if action.shape != (8,):
                        raise ValueError(
                            f"ACT returned invalid action: {action.shape}"
                        )

                    if not np.isfinite(action).all():
                        raise ValueError(
                            "ACT returned NaN or Inf"
                        )

                    send_packet(
                        conn,
                        {
                            "status": "ok",
                            "action": action.tolist(),
                        },
                    )

                except Exception:
                    error = traceback.format_exc()

                    print(
                        "\n[ACT WORKER][INFERENCE ERROR]\n"
                        + error,
                        file=sys.stderr,
                        flush=True,
                    )

                    send_packet(
                        conn,
                        {
                            "status": "error",
                            "error": error,
                        },
                    )

    finally:

        server.close()

        print("[ACT WORKER] Stopped.")


# ============================================================
# Isaac Lab 过程中使用的客户端
# ============================================================

class ACTInferenceClient:

    def __init__(
        self,
        checkpoint,
        inference_env="env_act",    #我的act环境名字
        port=18765,
    ):

        self.checkpoint = str(
            Path(checkpoint).resolve()
        )

        self.port = port
        self.process = None
        self.sock = None

        conda = (
            os.environ.get("CONDA_EXE")
            or shutil.which("conda")
        )

        if not conda:
            raise RuntimeError(
                "Cannot find conda executable."
            )

        script = Path(__file__).resolve()

        command = [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            inference_env,
            "python",
            str(script),
            "--inference-worker",
            "--checkpoint",
            self.checkpoint,
            "--port",
            str(port),
        ]

        print("[MAIN] Starting ACT inference worker:")
        print(" ".join(command))

        self.process = subprocess.Popen(
            command,
            cwd=str(script.parents[2]),
        )

        self._connect()

    def _connect(self):

        for _ in range(360):

            if self.process.poll() is not None:
                raise RuntimeError(
                    "ACT worker exited before connection."
                )

            try:

                sock = socket.create_connection(
                    ("127.0.0.1", self.port),
                    timeout=2,
                )

                sock.settimeout(120)

                response = recv_packet(sock)

                if response.get("status") != "ready":
                    sock.close()
                    raise RuntimeError(
                        f"Worker startup failed: {response}"
                    )

                self.sock = sock

                print(
                    "[MAIN] ACT worker ready:",
                    response,
                )

                return

            except (ConnectionRefusedError, OSError):
                time.sleep(0.5)

        raise TimeoutError(
            "ACT inference worker startup timed out."
        )

    def reset(self):

        send_packet(
            self.sock,
            {"cmd": "reset"},
        )

        response = recv_packet(self.sock)

        if response["status"] != "reset_ok":
            raise RuntimeError(response)

    def infer(self, state, wrist, front):

        send_packet(
            self.sock,
            {
                "cmd": "infer",
                "observation": {
                    "state": state,
                    "wrist": wrist,
                    "front": front,
                },
            },
        )

        response = recv_packet(self.sock)

        if response["status"] != "ok":
            raise RuntimeError(
                response.get("error", str(response))
            )

        return response["action"]

    def close(self):

        if self.sock is not None:

            try:
                send_packet(
                    self.sock,
                    {"cmd": "close"},
                )

                recv_packet(self.sock)

            except Exception:
                pass

            self.sock.close()
            self.sock = None

        if self.process is not None:

            try:
                self.process.wait(timeout=10)

            except subprocess.TimeoutExpired:
                self.process.terminate()


# ============================================================
# Isaac Lab helpers
# ============================================================

def get_camera_rgb(camera):

    import numpy as np

    rgb = camera.data.output["rgb"][0]

    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()

    rgb = np.asarray(rgb)

    if rgb.shape != (180, 320, 3):
        raise ValueError(
            f"Unexpected camera image shape: {rgb.shape}"
        )

    if rgb.dtype != np.uint8:

        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb * 255.0, 0, 255).astype(
                np.uint8
            )
        else:
            raise TypeError(
                f"Unexpected camera dtype: {rgb.dtype}"
            )

    return np.ascontiguousarray(rgb)


def get_tcp_hold_action(env):

    import torch

    ee_sensor = env.unwrapped.scene["ee_frame"]

    pos = (
        ee_sensor.data.target_pos_w[..., 0, :].clone()
        - env.unwrapped.scene.env_origins
    )

    quat = (
        ee_sensor.data.target_quat_w[..., 0, :].clone()
    )

    gripper = torch.ones(
        (env.unwrapped.num_envs, 1),
        device=env.unwrapped.device,
    )

    return torch.cat(
        [pos, quat, gripper],
        dim=-1,
    )


def get_ee_pose(env):
    """Return measured EE target-frame pose in env-local coordinates.

    FrameTransformer's target_pos_w/target_quat_w are the current measured
    target-frame pose in world coordinates, not the commanded ACT action.
    Quaternion order is WXYZ, matching the task action format.
    """
    import numpy as np

    ee_sensor = env.unwrapped.scene["ee_frame"]
    env_origin = env.unwrapped.scene.env_origins[0]

    position = (
        ee_sensor.data.target_pos_w[0, 0] - env_origin
    ).detach().cpu().numpy().astype(np.float32)

    quaternion = (
        ee_sensor.data.target_quat_w[0, 0]
    ).detach().cpu().numpy().astype(np.float32)

    return position, quaternion


def get_observation(env):

    import numpy as np

    scene = env.unwrapped.scene
    robot = scene["robot"]

    # Joint Position: 12
    qpos = (
        robot.data.joint_pos[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # Joint Velocity: 12
    qvel = (
        robot.data.joint_vel[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # EE Pose: 7
    # xyz + quaternion WXYZ
    ee_position, ee_quaternion = get_ee_pose(env)

    ee_pose = np.concatenate([
        ee_position,
        ee_quaternion
    ]).astype(np.float32)

    # The recorder stores both positions in the same env-local coordinates.
    object_position = (
        scene["object"].data.root_pos_w[0] - scene.env_origins[0]
    ).detach().cpu().numpy().astype(np.float32)
    state = build_state(qpos, qvel, ee_pose, object_position)

    if state.shape != (34,):
        raise ValueError(
            f"Invalid observation state: {state.shape}"
        )

    wrist = get_camera_rgb(
        scene["wrist_camera"]
    )

    front = get_camera_rgb(
        scene["front_camera"]
    )

    return state, wrist, front


def get_object_state(env):

    import numpy as np

    obj = env.unwrapped.scene["object"]

    origins = env.unwrapped.scene.env_origins

    position = (
        obj.data.root_pos_w[0] - origins[0]
    ).detach().cpu().numpy()

    velocity = None

    if hasattr(obj.data, "root_lin_vel_w"):

        velocity = (
            obj.data.root_lin_vel_w[0]
            .detach()
            .cpu()
            .numpy()
        )

    elif hasattr(obj.data, "root_vel_w"):

        velocity = (
            obj.data.root_vel_w[0, :3]
            .detach()
            .cpu()
            .numpy()
        )

    return position.astype(np.float32), velocity


def sanitize_action(
    raw_action,
    previous_position,
    previous_quaternion,
    policy_dt,
    max_ee_speed=0.5,
    workspace_x_max=0.52,
):
    """Apply workspace and commanded-target step limits to an ACT action.

    The 0.5 m/s value limits the distance between consecutive Cartesian
    command targets. It is not a measurement or hard guarantee of physical
    end-effector velocity.
    """
    import numpy as np

    action = np.asarray(raw_action, dtype=np.float32).reshape(-1).copy()

    if action.shape != (8,):
        raise ValueError(f"Expected action (8,), got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("Non-finite ACT action")

    workspace_min = np.array(
        [-0.02, -0.35, 0.005],
        dtype=np.float32,
    )
    workspace_max = np.array(
        [workspace_x_max, 0.30, 0.32],
        dtype=np.float32,
    )

    raw_position = action[:3].copy()
    position = np.clip(raw_position, workspace_min, workspace_max)
    workspace_clipped = not np.allclose(
        raw_position, position, atol=1e-6
    )

    # Preserve the 0.5 m/s demonstrated Cartesian target-step limit.
    delta = position - np.asarray(previous_position, dtype=np.float32)
    distance = float(np.linalg.norm(delta))
    max_step = float(max_ee_speed) * float(policy_dt)
    speed_limited = False

    if distance > max_step and distance > 1e-8:
        position = (
            np.asarray(previous_position, dtype=np.float32)
            + delta * (max_step / distance)
        )
        speed_limited = True

    # Quaternion order remains WXYZ.
    quaternion = action[3:7].copy()
    quat_norm = float(np.linalg.norm(quaternion))
    quaternion_adjusted = False

    if quat_norm < 1e-6:
        quaternion = np.asarray(
            previous_quaternion, dtype=np.float32
        ).copy()
        quaternion_adjusted = True
    else:
        quaternion = quaternion / quat_norm

    # q and -q represent the same rotation; keep sign continuous.
    if np.dot(quaternion, previous_quaternion) < 0:
        quaternion = -quaternion
        quaternion_adjusted = True

    # Original gripper encoding: +1 = OPEN, -1 = CLOSE.
    gripper = 1.0 if action[7] >= 0.0 else -1.0

    executed = np.concatenate(
        [position, quaternion, np.array([gripper], dtype=np.float32)]
    ).astype(np.float32)

    flags = {
        "workspace_clipped": bool(workspace_clipped),
        "speed_limited": bool(speed_limited),
        "quaternion_adjusted": bool(quaternion_adjusted),
        "action_clipped": bool(
            workspace_clipped or speed_limited or quaternion_adjusted
        ),
    }
    return executed, flags


def check_done(terminated, time_outs):

    def any_value(value):

        if hasattr(value, "any"):
            result = value.any()

            if hasattr(result, "item"):
                return bool(result.item())

            return bool(result)

        return bool(value)

    return (
        any_value(terminated),
        any_value(time_outs),
    )


# ============================================================
# Main simulation evaluation
# ============================================================

def run_isaaclab(args, simulation_app):

    import numpy as np
    import torch
    import gymnasium as gym

    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    # Import the real Lebai task registration.
    import act_manipulation.tasks.manager_based.lift.config.lebai_lm3  # noqa: F401

    from act_manipulation.tasks.manager_based.lift.lift_env_cfg import (
        LiftEnvCfg,
    )

    checkpoint = args.checkpoint.expanduser().resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    env_cfg: LiftEnvCfg = parse_env_cfg(
        ENV_ID,
        device=args.device,
        num_envs=1,
        use_fabric=not args.disable_fabric,
    )

    # Keep GUI and camera rendering enabled.
    env_cfg.rerender_on_reset = True

    env = gym.make(
        ENV_ID,
        cfg=env_cfg,
    )

    client = None

    try:

        env.reset()

        unwrapped = env.unwrapped

        if unwrapped.num_envs != 1:
            raise RuntimeError(
                "ACT evaluation currently requires num_envs=1."
            )

        action_shape = unwrapped.action_space.shape

        if action_shape[-1] != 8:
            raise RuntimeError(
                f"Expected 8D action space, got {action_shape}"
            )

        robot = unwrapped.scene["robot"]

        actual_joint_names = list(robot.joint_names)

        if actual_joint_names != JOINT_NAMES:
            raise RuntimeError(
                "Joint order mismatch!\n"
                f"Expected: {JOINT_NAMES}\n"
                f"Actual:   {actual_joint_names}"
            )

        print("[INFO] Joint order verified.")
        print("[INFO] Action space:", action_shape)
        print("[INFO] Sim dt:", env_cfg.sim.dt)
        print("[INFO] Decimation:", env_cfg.decimation)
        print("[INFO] Workspace X max:", args.workspace_x_max, "m")
        print("[INFO] Cartesian target-step limit:", args.max_ee_speed, "m/s")

        control_dt = (
            env_cfg.sim.dt * env_cfg.decimation
        )

        policy_dt = 1.0 / args.policy_fps

        sim_steps_per_action = round(
            policy_dt / control_dt
        )

        if not np.isclose(
            sim_steps_per_action * control_dt,
            policy_dt,
            atol=1e-5,
        ):
            raise RuntimeError(
                "Policy FPS is not aligned with environment control dt."
            )

        print(
            "[INFO] Policy frequency:",
            args.policy_fps,
            "Hz",
        )

        print(
            "[INFO] Sim steps per ACT action:",
            sim_steps_per_action,
        )

        # Start LeRobot inference worker.
        client = ACTInferenceClient(
            checkpoint=checkpoint,
            inference_env=args.inference_env,
            port=args.port,
        )

        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_path = output_dir / "summary.csv"
        steps_path = output_dir / "steps.csv"

        summary_fields = [
            "episode",
            "success",
            "termination_reason",
            "policy_steps",
            "sim_steps",
            "stable_steps",
            "initial_object_position",
            "final_object_position",
            "final_ee_position",
            "minimum_ee_object_center_distance",
        ]

        step_fields = [
            "episode",
            "policy_step",
            "raw_action",
            "executed_action",
            "raw_target_position",
            "executed_target_position",
            "actual_ee_position",
            "actual_ee_quaternion_wxyz",
            "ee_object_center_distance",
            "ee_raw_target_distance",
            "ee_executed_target_distance",
            "workspace_clipped",
            "speed_limited",
            "quaternion_adjusted",
            "action_clipped",
            "object_position",
            "object_speed_displacement",
            "object_speed_physx",
            "gripper",
            "inside_tray",
        ]

        summary_file = summary_path.open(
            "w",
            newline="",
        )

        steps_file = steps_path.open(
            "w",
            newline="",
        )

        summary_writer = csv.DictWriter(
            summary_file,
            fieldnames=summary_fields,
        )

        steps_writer = csv.DictWriter(
            steps_file,
            fieldnames=step_fields,
        )

        summary_writer.writeheader()
        steps_writer.writeheader()

        total_success = 0
        total_episodes = 0

        # Approximate geometric success region.
        # Tray center comes from lift_env_cfg.py:
        # (0.0, -0.3, 0.03)
        #
        # Half extents are configurable because the
        # exact usable inner dimensions depend on USD geometry.

        tray_center = np.array(
            [
                args.tray_center_x,
                args.tray_center_y,
            ],
            dtype=np.float32,
        )

        try:

            for episode in range(args.num_episodes):

                if not simulation_app.is_running():
                    break

                print(
                    f"\n========== EPISODE "
                    f"{episode + 1}/{args.num_episodes} =========="
                )

                # Reset simulator and ACT action queue.
                if episode > 0:
                    env.reset()
                
                print("[DEBUG] Before ACT reset", flush=True)
                client.reset()
                print("[DEBUG] ACT reset OK", flush=True)
                
                # Warm up cameras at 10 Hz.
                hold_action = get_tcp_hold_action(env)

                for _ in range(sim_steps_per_action):
                    env.step(hold_action)

                ee_sensor = unwrapped.scene["ee_frame"]

                previous_position = (
                    ee_sensor.data.target_pos_w[0, 0]
                    - unwrapped.scene.env_origins[0]
                ).detach().cpu().numpy().copy()

                previous_quaternion = (
                    ee_sensor.data.target_quat_w[0, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()
                )

                initial_object_position, _ = (
                    get_object_state(env)
                )

                last_object_position = (
                    initial_object_position.copy()
                )

                stable_count = 0
                success = False
                reason = "max_policy_steps"
                minimum_ee_object_center_distance = float("inf")

                policy_steps = 0
                sim_steps = sim_steps_per_action

                # Leave a small margin before the
                # environment's 10-second timeout.
                max_steps = min(
                    args.max_policy_steps,
                    int(
                        max(
                            0.0,
                            env_cfg.episode_length_s
                            - policy_dt
                            - 0.2,
                        )
                        * args.policy_fps
                    ),
                )

                for policy_step in range(max_steps):

                    if not simulation_app.is_running():
                        reason = "simulation_closed"
                        break

                    # ------------------------------------------------
                    # 1. Current observation
                    # ------------------------------------------------

                    print("[DEBUG] Reading observation", flush=True)

                    state, wrist, front = get_observation(env)

                    print(
                        "[DEBUG] Observation OK",
                        state.shape,
                        wrist.shape,
                        front.shape,
                        flush=True,
                    )

                    # ------------------------------------------------
                    # 2. ACT inference
                    # ------------------------------------------------

                    print("[DEBUG] Calling ACT inference", flush=True)

                    raw_action = client.infer(
                        state=state,
                        wrist=wrist,
                        front=front,
                    )

                    print(
                        "[DEBUG] ACT raw action:",
                        raw_action,
                        flush=True,
                    )

                    # ------------------------------------------------
                    # 3. Safety filtering
                    # ------------------------------------------------

                    raw_action_array = np.asarray(
                        raw_action, dtype=np.float32
                    ).reshape(8)

                    executed_action, clip_flags = sanitize_action(
                        raw_action=raw_action_array,
                        previous_position=previous_position,
                        previous_quaternion=previous_quaternion,
                        policy_dt=policy_dt,
                        max_ee_speed=args.max_ee_speed,
                        workspace_x_max=args.workspace_x_max,
                    )

                    previous_position = (
                        executed_action[:3].copy()
                    )

                    previous_quaternion = (
                        executed_action[3:7].copy()
                    )

                    # Exact original environment action:
                    # [x,y,z,qw,qx,qy,qz,gripper]
                    action_tensor = torch.as_tensor(
                        executed_action,
                        dtype=torch.float32,
                        device=unwrapped.device,
                    ).view(1, 8)

                    # ------------------------------------------------
                    # 4. Execute at 50 Hz while ACT updates at 10 Hz
                    # ------------------------------------------------

                    terminated_flag = False
                    timeout_flag = False

                    for _ in range(sim_steps_per_action):

                        print("[DEBUG] Executing ACT action", flush=True)
                        _, _, terminated, time_outs, _ = (
                            env.step(action_tensor)
                        )

                        sim_steps += 1

                        terminated_flag, timeout_flag = (
                            check_done(
                                terminated,
                                time_outs,
                            )
                        )

                        if terminated_flag or timeout_flag:
                            break

                    policy_steps += 1

                    if terminated_flag:
                        reason = "environment_terminated"
                        break

                    if timeout_flag:
                        reason = "environment_timeout"
                        break

                    # ------------------------------------------------
                    # 5. Geometric placement evaluation
                    # ------------------------------------------------

                    object_position, object_velocity = get_object_state(env)

                    # Position-difference speed over one policy interval.
                    # This is kept separate from PhysX's instantaneous velocity.
                    object_speed_displacement = float(
                        np.linalg.norm(
                            object_position - last_object_position
                        ) / policy_dt
                    )

                    object_speed_physx = (
                        float(np.linalg.norm(object_velocity))
                        if object_velocity is not None
                        else float("nan")
                    )

                    last_object_position = object_position.copy()

                    # Measure actual EE pose after the environment steps.
                    actual_ee_position, actual_ee_quaternion = get_ee_pose(env)
                    # This is EE-frame-origin to object-root-position distance,
                    # not the physical surface-to-surface contact gap.
                    ee_object_center_distance = float(
                        np.linalg.norm(actual_ee_position - object_position)
                    )
                    ee_raw_target_distance = float(
                        np.linalg.norm(
                            actual_ee_position - raw_action_array[:3]
                        )
                    )
                    ee_executed_target_distance = float(
                        np.linalg.norm(
                            actual_ee_position - executed_action[:3]
                        )
                    )
                    minimum_ee_object_center_distance = min(
                        minimum_ee_object_center_distance, ee_object_center_distance
                    )

                    inside_tray = (
                        abs(
                            object_position[0]
                            - tray_center[0]
                        ) <= args.tray_half_x
                        and
                        abs(
                            object_position[1]
                            - tray_center[1]
                        ) <= args.tray_half_y
                        and
                        args.object_z_min
                        <= object_position[2]
                        <= args.object_z_max
                    )

                    gripper_open = (
                        executed_action[7] > 0
                    )

                    settled = (
                        inside_tray
                        and gripper_open
                        and object_speed_displacement < args.object_speed_threshold
                    )

                    if settled:
                        stable_count += 1
                    else:
                        stable_count = 0

                    # ------------------------------------------------
                    # 6. Log
                    # ------------------------------------------------

                    steps_writer.writerow(
                        {
                            "episode": episode + 1,
                            "policy_step": policy_step + 1,
                            "raw_action": json.dumps(
                                raw_action_array.tolist()
                            ),
                            "executed_action": json.dumps(
                                executed_action.tolist()
                            ),
                            "raw_target_position": json.dumps(
                                raw_action_array[:3].tolist()
                            ),
                            "executed_target_position": json.dumps(
                                executed_action[:3].tolist()
                            ),
                            "actual_ee_position": json.dumps(
                                actual_ee_position.tolist()
                            ),
                            "actual_ee_quaternion_wxyz": json.dumps(
                                actual_ee_quaternion.tolist()
                            ),
                            "ee_object_center_distance": ee_object_center_distance,
                            "ee_raw_target_distance": ee_raw_target_distance,
                            "ee_executed_target_distance": ee_executed_target_distance,
                            "workspace_clipped": clip_flags["workspace_clipped"],
                            "speed_limited": clip_flags["speed_limited"],
                            "quaternion_adjusted": clip_flags["quaternion_adjusted"],
                            "action_clipped": clip_flags["action_clipped"],
                            "object_position": json.dumps(
                                object_position.tolist()
                            ),
                            "object_speed_displacement": object_speed_displacement,
                            "object_speed_physx": object_speed_physx,
                            "gripper": float(executed_action[7]),
                            "inside_tray": inside_tray,
                        }
                    )

                    steps_file.flush()

                    if policy_step % 10 == 0:

                        print(
                            f"[EP {episode + 1}] "
                            f"step={policy_step + 1:03d} "
                            f"object={np.round(object_position, 3)} "
                            f"ee={np.round(actual_ee_position, 3)} "
                            f"ee_obj_center_dist={ee_object_center_distance:.3f}m "
                            f"obj_v_disp={object_speed_displacement:.3f}m/s "
                            f"gripper={executed_action[7]:+.0f} "
                            f"clip={clip_flags['action_clipped']} "
                            f"stable={stable_count}"
                        )

                    # Five consecutive 10Hz frames:
                    # object remains in tray, stationary,
                    # and gripper command is OPEN.
                    if stable_count >= args.success_stable_steps:

                        success = True
                        reason = "object_settled_in_tray"

                        print(
                            "[SUCCESS] Geometric placement condition met."
                        )

                        break

                # ----------------------------------------------------
                # 7. Episode summary
                # ----------------------------------------------------

                final_object_position, _ = get_object_state(env)
                final_ee_position, _ = get_ee_pose(env)

                total_episodes += 1

                if success:
                    total_success += 1

                summary_writer.writerow(
                    {
                        "episode": episode + 1,
                        "success": int(success),
                        "termination_reason": reason,
                        "policy_steps": policy_steps,
                        "sim_steps": sim_steps,
                        "stable_steps": stable_count,
                        "initial_object_position": json.dumps(
                            initial_object_position.tolist()
                        ),
                        "final_object_position": json.dumps(
                            final_object_position.tolist()
                        ),
                        "final_ee_position": json.dumps(
                            final_ee_position.tolist()
                        ),
                        "minimum_ee_object_center_distance": (
                            minimum_ee_object_center_distance
                            if np.isfinite(minimum_ee_object_center_distance)
                            else ""
                        ),
                    }
                )

                summary_file.flush()

                print(
                    f"[EPISODE RESULT] "
                    f"success={success} "
                    f"reason={reason} "
                    f"steps={policy_steps}"
                )

                print(
                    f"[OBJECT FINAL] "
                    f"{np.round(final_object_position, 4)}"
                )

        finally:

            summary_file.close()
            steps_file.close()

        print("\n========== ACT EVALUATION COMPLETE ==========")

        print("Episodes:", total_episodes)

        print("Success:", total_success)

        if total_episodes:
            print(
                "Success rate:",
                f"{100.0 * total_success / total_episodes:.1f}%",
            )

        print("Summary:", summary_path)
        print("Step logs:", steps_path)

    finally:

        if client is not None:
            client.close()

        env.close()


# ============================================================
# Entrypoint
# ============================================================

def main():

    # The same file runs under env_act when invoked
    # with --inference-worker.
    if "--inference-worker" in sys.argv:

        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--inference-worker",
            action="store_true",
        )

        parser.add_argument(
            "--checkpoint",
            type=Path,
            required=True,
        )

        parser.add_argument(
            "--port",
            type=int,
            default=18765,
        )

        worker_args = parser.parse_args()

        run_inference_worker(worker_args)

        return

    # Only the Isaac Lab process imports AppLauncher.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Evaluate ACT on Lebai LM3 in Isaac Lab."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/train/act_lebai_lm3_v7/"
            "checkpoints/060000/pretrained_model"
        ),
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,#设置数量
    )

    parser.add_argument(
        "--max-policy-steps",
        type=int,
        default=95,
    )

    parser.add_argument(
        "--policy-fps",
        type=int,
        default=10, #原本是10
    )

    parser.add_argument(
        "--max-ee-speed",
        type=float,
        default=0.5,#原本是0.5
    )

    parser.add_argument(
        "--workspace-x-max",
        type=float,
        default=0.52,  #原本是0.52
        help="Maximum allowed Cartesian X target in meters (default: 0.52).",
    )

    parser.add_argument(
        "--inference-env",
        type=str,
        default="env_act",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=18765,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/eval/act_lebai_lm3_v7"
        ),
    )

    parser.add_argument(
        "--disable-fabric",
        action="store_true",
    )

    # Tray geometry: approximate inner region.
    parser.add_argument("--tray-center-x", type=float, default=0.0)
    parser.add_argument("--tray-center-y", type=float, default=-0.3)
    parser.add_argument("--tray-half-x", type=float, default=0.12)
    parser.add_argument("--tray-half-y", type=float, default=0.12)

    parser.add_argument("--object-z-min", type=float, default=0.025)
    parser.add_argument("--object-z-max", type=float, default=0.20)

    parser.add_argument(
        "--object-speed-threshold",
        type=float,
        default=0.04,
    )

    parser.add_argument(
        "--success-stable-steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
    )

    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()

    if args.num_envs != 1:
        raise ValueError(
            "ACT evaluation currently supports exactly one environment."
        )

    if args.num_episodes < 1:
        raise ValueError("--num-episodes must be positive.")

    if args.policy_fps < 1:
        raise ValueError("--policy-fps must be positive.")

    if args.max_ee_speed <= 0:
        raise ValueError("--max-ee-speed must be positive.")

    if args.workspace_x_max <= -0.02:
        raise ValueError("--workspace-x-max must be greater than -0.02 m.")

    # Camera rendering is mandatory.
    args.enable_cameras = True

    app_launcher = AppLauncher(args)

    try:
        run_isaaclab(
            args,
            app_launcher.app,
        )

    except BaseException:
        print(
            "\n[MAIN][FATAL ERROR]",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise

    finally:
        print(
            "[MAIN] Closing Isaac Sim",
            flush=True,
        )
        app_launcher.app.close()


if __name__ == "__main__":
    main()

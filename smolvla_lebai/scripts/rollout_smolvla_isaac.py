#!/usr/bin/env python3
"""SmolVLA worker plus existing ACT v2 Isaac Lab simulator/control bridge.

Run this file with Isaac Lab's Python. It starts a separate SmolVLA Python
process and reuses the existing absolute IK, camera, gripper and safety path.
Only use with simulation; this file has no real-robot interface.
"""
from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.act import eval_act_isaaclab_v2 as bridge

PYTHON = ROOT / ".conda/smolvla_lebai/bin/python"
TASK = "Pick up the cube and place it into the tray."
CAPTURE_STATE = {"episode": 0, "step": 0}


def worker(args: argparse.Namespace) -> None:
    import torch
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.utils import build_inference_frame

    checkpoint = args.checkpoint.resolve()
    model = SmolVLAPolicy.from_pretrained(checkpoint).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=model.config, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    assert model.config.input_features["observation.state"].shape == (31,)
    assert model.config.output_features["action"].shape == (8,)
    assert len([key for key in model.config.input_features if key.startswith("observation.images.")]) == 2
    print("[SMOLVLA] checkpoint", checkpoint, "chunk_size", model.config.chunk_size,
          "n_action_steps", model.config.n_action_steps, flush=True)
    seed_base = os.environ.get("SMOLVLA_EVAL_SEED")
    reset_index = 0
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", args.port)); server.listen(1)
    try:
        conn, _ = server.accept()
        with conn:
            conn.settimeout(120)
            bridge.send_packet(conn, {"status": "ready", "device": "cuda",
                                      "chunk_size": model.config.chunk_size,
                                      "n_action_steps": model.config.n_action_steps})
            while True:
                request = bridge.recv_packet(conn)
                command = request["cmd"]
                if command == "close":
                    bridge.send_packet(conn, {"status": "closed"}); break
                if command == "reset":
                    if seed_base is not None:
                        seed = int(seed_base) + reset_index
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        np.random.seed(seed)
                    reset_index += 1
                    model.reset(); pre.reset(); post.reset()
                    bridge.send_packet(conn, {"status": "reset_ok"}); continue
                try:
                    if command != "infer":
                        raise ValueError(command)
                    raw = request["observation"]
                    state = np.asarray(raw["state"], dtype=np.float32)
                    wrist = np.asarray(raw["wrist"], dtype=np.uint8)
                    front = np.asarray(raw["front"], dtype=np.uint8)
                    if state.shape != (31,) or not np.isfinite(state).all():
                        raise ValueError("Invalid 31D state")
                    if wrist.shape != (180, 320, 3) or front.shape != (180, 320, 3):
                        raise ValueError("Invalid camera image")
                    values = {name: float(state[i]) for i, name in enumerate(bridge.STATE_NAMES)}
                    values.update({"front": front, "wrist": wrist})
                    frame = build_inference_frame(values, device="cuda", ds_features=bridge.DATASET_FEATURES, task=TASK)
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        action = post(model.select_action(pre(frame)))
                        torch.cuda.synchronize()
                    result = action.reshape(-1).detach().float().cpu().numpy()
                    if result.shape != (8,) or not np.isfinite(result).all():
                        raise ValueError("Nonfinite or malformed action")
                    bridge.send_packet(conn, {"status": "ok", "action": result.tolist(),
                                              "inference_s": time.perf_counter() - t0})
                except Exception:
                    bridge.send_packet(conn, {"status": "error", "error": traceback.format_exc()})
    finally:
        server.close()


class SmolVLAInferenceClient(bridge.ACTInferenceClient):
    def __init__(self, checkpoint, inference_env=None, port=18765):
        self.checkpoint = str(Path(checkpoint).resolve())
        self.port = port
        self.sock = None
        self.episode = 0
        self.policy_step = 0
        log_path = os.environ.get("SMOLVLA_LATENCY_LOG")
        self.latency_path = Path(log_path) if log_path else None
        if self.latency_path is not None:
            self.latency_path.parent.mkdir(parents=True, exist_ok=True)
            if self.latency_path.exists():
                raise FileExistsError(f"Refusing to overwrite latency log: {self.latency_path}")
            with self.latency_path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["episode", "policy_step", "wall_time_s", "roundtrip_s", "worker_inference_s"])
        command = [str(PYTHON), str(Path(__file__).resolve()), "--inference-worker",
                   "--checkpoint", self.checkpoint, "--port", str(port)]
        self.process = subprocess.Popen(command, cwd=str(ROOT), env={**os.environ, "HF_HOME": str(ROOT / ".cache/huggingface")})
        self._connect()

    def reset(self):
        super().reset()
        self.episode += 1
        self.policy_step = 0
        CAPTURE_STATE["episode"] = self.episode
        CAPTURE_STATE["step"] = 0

    def infer(self, state, wrist, front):
        t0 = time.perf_counter()
        bridge.send_packet(self.sock, {"cmd": "infer", "observation":
                                   {"state": state, "wrist": wrist, "front": front}})
        response = bridge.recv_packet(self.sock)
        finish = time.perf_counter()
        if response.get("status") != "ok":
            raise RuntimeError(response.get("error", str(response)))
        self.policy_step += 1
        if self.latency_path is not None:
            with self.latency_path.open("a", newline="") as stream:
                csv.writer(stream).writerow([self.episode, self.policy_step, finish,
                                             finish - t0, response["inference_s"]])
        return response["action"]


def main() -> None:
    if "--inference-worker" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--inference-worker", action="store_true")
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--port", type=int, default=18765)
        worker(parser.parse_args())
        return
    if "--smol-dry-run" in sys.argv:
        sys.argv.remove("--smol-dry-run")
        dry_run()
        return
    if "--smol-seed" in sys.argv:
        index = sys.argv.index("--smol-seed")
        seed = int(sys.argv[index + 1])
        del sys.argv[index : index + 2]
        os.environ["SMOLVLA_EVAL_SEED"] = str(seed)
        original_run = bridge.run_isaaclab

        def seeded_run(args, simulation_app):
            import gymnasium as gym
            original_make = gym.make

            def seeded_make(*make_args, **make_kwargs):
                env = original_make(*make_args, **make_kwargs)
                original_reset = env.reset
                reset_index = 0

                def seeded_reset(*reset_args, **reset_kwargs):
                    nonlocal reset_index
                    reset_kwargs["seed"] = seed + reset_index
                    reset_index += 1
                    return original_reset(*reset_args, **reset_kwargs)

                env.reset = seeded_reset
                return env

            gym.make = seeded_make
            try:
                return original_run(args, simulation_app)
            finally:
                gym.make = original_make

        bridge.run_isaaclab = seeded_run
    if "--smol-capture-frames" in sys.argv:
        sys.argv.remove("--smol-capture-frames")
        from PIL import Image

        original_get_observation = bridge.get_observation
        capture_dir = ROOT / f"smolvla_lebai/figures/failure_seed{os.environ.get('SMOLVLA_EVAL_SEED', 'unfixed')}"
        capture_dir.mkdir(parents=True, exist_ok=True)

        def capture_observation(env):
            state, wrist, front = original_get_observation(env)
            CAPTURE_STATE["step"] += 1
            episode, step = CAPTURE_STATE["episode"], CAPTURE_STATE["step"]
            if episode == 1 and step in (1, 50, 95, 150, 200, 240):
                for camera, image in (("front", front), ("wrist", wrist)):
                    target = capture_dir / f"{camera}_step{step:03d}.png"
                    if target.exists():
                        raise FileExistsError(f"Refusing to overwrite {target}")
                    Image.fromarray(image).save(target)
            return state, wrist, front

        bridge.get_observation = capture_observation
    bridge.ACTInferenceClient = SmolVLAInferenceClient
    def option_value(name: str) -> str | None:
        for index, value in enumerate(sys.argv[1:], start=1):
            if value == name:
                return sys.argv[index + 1]
            if value.startswith(name + "="):
                return value.split("=", 1)[1]
        return None

    if option_value("--checkpoint") is None:
        sys.argv.extend(["--checkpoint", str(ROOT / "outputs/train/smolvla_lebai_1000/checkpoints/001000/pretrained_model")])
    if option_value("--policy-fps") is None:
        sys.argv.extend(["--policy-fps", "25"])
    if option_value("--output-dir") is None:
        sys.argv.extend(["--output-dir", str(ROOT / "outputs/eval/smolvla_isaac")])
    os.environ["SMOLVLA_LATENCY_LOG"] = str(Path(option_value("--output-dir")) / "latency.csv")
    bridge.main()


def dry_run() -> None:
    """Read one Isaac Lab observation and predict one action; never call env.step()."""
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/train/smolvla_lebai_1000/checkpoints/001000/pretrained_model")
    parser.add_argument("--port", type=int, default=18765)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    client = None
    env = None
    try:
        import gymnasium as gym
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        import act_manipulation.tasks.manager_based.lift.config.lebai_lm3  # noqa: F401
        cfg = parse_env_cfg(bridge.ENV_ID, device=args.device, num_envs=1, use_fabric=True)
        cfg.rerender_on_reset = True
        env = gym.make(bridge.ENV_ID, cfg=cfg)
        env.reset()
        state, wrist, front = bridge.get_observation(env)
        client = SmolVLAInferenceClient(args.checkpoint, port=args.port)
        client.reset()
        t0 = time.perf_counter()
        action = np.asarray(client.infer(state, wrist, front), dtype=np.float32)
        latency = time.perf_counter() - t0
        ee_position, ee_quaternion = bridge.get_ee_pose(env)
        safe, flags = bridge.sanitize_action(action, ee_position, ee_quaternion, 0.04, 0.5, 0.52)
        print({"status": "dry_run_no_env_step", "raw_action": action.tolist(),
               "safe_action": safe.tolist(), "safety_flags": flags, "roundtrip_latency_s": latency}, flush=True)
    finally:
        if client is not None:
            client.close()
        if env is not None:
            env.close()
        app_launcher.app.close()


if __name__ == "__main__":
    main()

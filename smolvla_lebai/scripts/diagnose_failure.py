#!/usr/bin/env python3
"""Recompute horizon, action, safety and phase diagnostics from recorded artifacts."""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
from scripts.act.eval_act_isaaclab_v2 import sanitize_action

BASE = ROOT / "datasets/lerobot/lebai_lm3_pick_place_25hz"
RAW = ROOT / "datasets/raw_25hz"
EVAL = ROOT / "outputs/eval"
FIG = ROOT / "smolvla_lebai/figures"
FIG.mkdir(parents=True, exist_ok=True)
STAGES = ["Rest", "Above", "Approach", "Grasp", "Lift", "Place", "Release", "Retreat", "Done"]


def angle_deg(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if min(na, nb) < 1e-8:
        return np.nan
    return float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(a / na, b / nb)), 0, 1))))


def load_steps(name):
    with (EVAL / name / "steps.csv").open() as stream:
        return list(csv.DictReader(stream))


def summarize(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "p50": float(np.median(a)), "p90": float(np.quantile(a, .9)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max())}


def main():
    manifest = json.loads((BASE / "split_manifest.json").read_text())
    source_ids = manifest["train_source_episode_ids"] + manifest["val_source_episode_ids"]
    lengths, starts, at95 = [], defaultdict(list), Counter()
    expert_flags = Counter(); expert_n = 0; expert_qnorm = []; expert_step = []; expert_xyz_filter = []
    all_episodes = {}
    for source_id in source_ids:
        with h5py.File(RAW / f"episode_{source_id:06d}.hdf5", "r") as h:
            fsm = h["fsm_state"][:].astype(int)
            action = h["action"][:].astype(np.float32)
            ee = h["observations/ee_pose"][:].astype(np.float32)
            obj = h["object_position"][:].astype(np.float32)
            lengths.append(len(fsm)); at95[int(fsm[94])] += 1
            all_episodes[source_id] = fsm
            for stage in range(9):
                indices = np.flatnonzero(fsm == stage)
                if len(indices): starts[stage].append(int(indices[0]))
            previous_pos, previous_q = ee[0, :3], ee[0, 3:7]
            for step, a in enumerate(action):
                filtered, flags = sanitize_action(a, previous_pos, previous_q, .04, .5, .52)
                expert_flags.update(k for k, v in flags.items() if v)
                expert_n += 1
                expert_xyz_filter.append(float(np.linalg.norm(a[:3] - filtered[:3])))
                expert_qnorm.append(float(np.linalg.norm(a[3:7])))
                if step:
                    expert_step.append(float(np.linalg.norm(a[:3] - action[step - 1, :3])))
                previous_pos, previous_q = filtered[:3], filtered[3:7]

    stage_table = [{"stage": STAGES[k], "first_frame_p50": float(np.median(v)),
                    "first_frame_p90": float(np.quantile(v, .9)),
                    "first_time_p50_s": float(np.median(v) / 25)} for k, v in sorted(starts.items())]
    rows = json.loads((EVAL / "smolvla_lebai/val_fsm_windows.json").read_text())["rows"]
    phase = defaultdict(lambda: {"xyz": [], "angle": [], "gripper": [], "count": 0})
    chunk = defaultdict(lambda: {"xyz": [], "angle": []})
    pred_norms, pred_qsmall, original_alignment = [], 0, []
    actual_switches = predicted_switches = switch_matches = 0
    train_mean = np.array(json.loads((BASE / "train/meta/stats.json").read_text())["action"]["mean"])
    baselines = {"constant_train_mean": defaultdict(list), "state_hold_open": defaultdict(list)}
    dataset = ROOT / "datasets/lerobot/lebai_lm3_pick_place_25hz_smolvla31/val/data/chunk-000/file-000.parquet"
    import pyarrow.parquet as pq
    table = pq.read_table(dataset, columns=["observation.state", "action", "episode_index", "frame_index"])
    ep_col = np.asarray(table["episode_index"]); frame_col = np.asarray(table["frame_index"])
    state_col = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action_col = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    index_by_frame = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(ep_col, frame_col))}
    for i, row in enumerate(rows):
        ep, frame = row["episode"], row["frame"]
        source_id = manifest["val_dataset_episode_to_source"][ep]
        stage = int(all_episodes[source_id][frame])
        expert, pred = np.array(row["expert"]), np.array(row["predicted"])
        ref = index_by_frame[(ep, frame)]
        assert np.allclose(expert, action_col[ref], atol=1e-6)
        with h5py.File(RAW / f"episode_{source_id:06d}.hdf5", "r") as h:
            original_alignment.append(float(np.max(np.abs(expert - h["action"][frame]))))
        p = phase[stage]; p["xyz"].append(row["xyz_mae_m"]); p["angle"].append(row["quat_angle_deg"])
        p["gripper"].append(row["gripper_correct"]); p["count"] += 1
        j = row["chunk_offset"]
        chunk[j]["xyz"].append(row["xyz_mae_m"]); chunk[j]["angle"].append(row["quat_angle_deg"])
        norm = float(np.linalg.norm(pred[3:7])); pred_norms.append(norm); pred_qsmall += norm < .1
        for name, baseline in (("constant_train_mean", train_mean),
                               ("state_hold_open", np.r_[state_col[ref, 24:31], 1.0])):
            baselines[name]["xyz"].append(float(np.mean(np.abs(baseline[:3] - expert[:3]))))
            baselines[name]["angle"].append(angle_deg(baseline[3:7], expert[3:7]))
            baselines[name]["gripper"].append(bool((baseline[7] >= 0) == (expert[7] >= 0)))
        if i and rows[i - 1]["episode"] == ep and frame == rows[i - 1]["frame"] + 1:
            true_change = (expert[7] >= 0) != (rows[i - 1]["expert"][7] >= 0)
            pred_change = (pred[7] >= 0) != (rows[i - 1]["predicted"][7] >= 0)
            actual_switches += true_change; predicted_switches += pred_change; switch_matches += true_change and pred_change

    sim = load_steps("smolvla_isaac_20")
    sim_flags = Counter(); xyz_filter = []; model_raw_jump = []; model_exe_jump = []
    replan_jump = []; queue_jump = []; distances = []; near = 0
    grouped = defaultdict(list)
    for row in sim:
        grouped[int(row["episode"])].append(row)
        sim_flags.update(k for k in ("workspace_clipped", "speed_limited", "quaternion_adjusted", "action_clipped") if row[k] == "True")
        raw = np.array(json.loads(row["raw_action"])); exe = np.array(json.loads(row["executed_action"]))
        xyz_filter.append(float(np.linalg.norm(raw[:3] - exe[:3])))
        distances.append(float(row["ee_object_center_distance"]))
        near += float(row["ee_object_center_distance"]) < .05
    for episode_rows in grouped.values():
        for prev, now in zip(episode_rows[:-1], episode_rows[1:]):
            ra, rb = np.array(json.loads(prev["raw_action"])), np.array(json.loads(now["raw_action"]))
            ea, eb = np.array(json.loads(prev["executed_action"])), np.array(json.loads(now["executed_action"]))
            jump = float(np.linalg.norm(rb[:3] - ra[:3])); model_raw_jump.append(jump)
            model_exe_jump.append(float(np.linalg.norm(eb[:3] - ea[:3])))
            (replan_jump if int(now["policy_step"]) % 10 == 1 else queue_jump).append(jump)

    metrics = {
        "expert_frames": expert_n, "episode_length_frames": summarize(lengths),
        "episode_length_mean_s": float(np.mean(lengths) / 25),
        "expert_fsm_at_frame95": {str(k): int(v) for k, v in at95.items()},
        "stage_first_occurrence": stage_table,
        "expert_filter_fraction": {k: expert_flags[k] / expert_n for k in ("workspace_clipped", "speed_limited", "quaternion_adjusted", "action_clipped")},
        "expert_raw_to_executed_xyz_distance_m": summarize(expert_xyz_filter),
        "expert_xyz_modified_over_1mm_fraction": float(np.mean(np.asarray(expert_xyz_filter) > .001)),
        "expert_xyz_step_m": summarize(expert_step), "expert_quat_norm": summarize(expert_qnorm),
        "model_filter_fraction": {k: sim_flags[k] / len(sim) for k in ("workspace_clipped", "speed_limited", "quaternion_adjusted", "action_clipped")},
        "model_raw_to_executed_xyz_distance_m": summarize(xyz_filter),
        "model_xyz_modified_over_1mm_fraction": float(np.mean(np.asarray(xyz_filter) > .001)),
        "model_raw_xyz_jump_m": summarize(model_raw_jump), "model_executed_xyz_jump_m": summarize(model_exe_jump),
        "model_replan_raw_xyz_jump_m": summarize(replan_jump), "model_queue_raw_xyz_jump_m": summarize(queue_jump),
        "model_raw_quat_norm_val_fsm": summarize(pred_norms), "model_quat_norm_below_0p1": pred_qsmall,
        "model_ee_object_distance_m": summarize(distances), "model_steps_ee_object_below_0p05m": near,
        "expert_action_alignment_max_abs_diff": max(original_alignment),
        "gripper_switches_in_sample": {"expert": actual_switches, "predicted": predicted_switches, "matching": switch_matches},
        "phase_metrics": {STAGES[k]: {"count": v["count"], "xyz_mae_m": float(np.mean(v["xyz"])),
                                      "quat_angle_deg": float(np.mean(v["angle"])),
                                      "gripper_accuracy": float(np.mean(v["gripper"]))} for k, v in sorted(phase.items())},
        "chunk_offset_metrics": {str(k): {"xyz_mae_m": float(np.mean(v["xyz"])),
                                          "quat_angle_deg": float(np.mean(v["angle"]))} for k, v in sorted(chunk.items())},
        "baselines": {name: {key: float(np.nanmean(values)) for key, values in measure.items()}
                      for name, measure in baselines.items()},
    }
    transitions = json.loads((EVAL / "smolvla_lebai/val_transition_windows.json").read_text())["rows"]
    transition_groups = defaultdict(list)
    for row in transitions:
        transition_groups[(row["episode"], row["window_stage"])].append(row)
    transition_metrics = {}
    for stage in (3, 6):
        delays, missing = [], 0
        correct = []
        for episode in range(40):
            window = transition_groups[(episode, stage)]
            true = np.array([row["expert"][7] >= 0 for row in window])
            predicted = np.array([row["predicted"][7] >= 0 for row in window])
            desired = stage == 6
            true_index = int(np.flatnonzero(true == desired)[0])
            predicted_indices = np.flatnonzero(predicted == desired)
            if len(predicted_indices):
                delays.append(int(predicted_indices[0]) - true_index)
            else:
                missing += 1
            correct.extend((true == predicted).tolist())
        transition_metrics[STAGES[stage]] = {
            "windows": 40, "missing_desired_sign": missing,
            "median_first_desired_sign_offset_frames": float(np.median(delays)),
            "early_windows": int(sum(delay < 0 for delay in delays)),
            "exact_windows": int(sum(delay == 0 for delay in delays)),
            "late_windows": int(sum(delay > 0 for delay in delays)),
            "framewise_gripper_accuracy": float(np.mean(correct)),
        }
    metrics["transition_metrics"] = transition_metrics
    train = json.loads((EVAL / "smolvla_lebai/train_fsm_windows.json").read_text())
    val = json.loads((EVAL / "smolvla_lebai/val_fsm_windows.json").read_text())
    metrics["train_val_fsm_windows"] = {
        split: {key: data[key] for key in ("frames", "episodes", "xyz_mae_m", "quat_angle_mean_deg", "gripper_accuracy")}
        for split, data in (("train", train), ("val", val))
    }
    out = EVAL / "smolvla_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    encoder = lambda value: value.item() if hasattr(value, "item") else str(value)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=encoder) + "\n")

    # Figures use measured traces only; GitHub-compatible assets live beside the docs.
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    sample = [r for r in rows if r["episode"] == 0]
    for axis, dim, name in zip(axes, range(3), ("X", "Y", "Z")):
        axis.plot([r["frame"] for r in sample], [r["expert"][dim] for r in sample], "o-", label="Expert", ms=3)
        axis.plot([r["frame"] for r in sample], [r["predicted"][dim] for r in sample], "o-", label="SmolVLA", ms=3)
        axis.set_ylabel(f"{name} (m)"); axis.grid(alpha=.2)
    axes[0].legend(); axes[-1].set_xlabel("Frame (episode 0; three sampled windows)")
    fig.tight_layout(); fig.savefig(FIG / "offline_xyz_val_ep0.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].bar(range(9), [metrics["phase_metrics"].get(STAGES[k], {}).get("xyz_mae_m", 0) for k in range(9)])
    axes[0].set_ylabel("XYZ MAE (m)")
    axes[1].bar(range(9), [metrics["phase_metrics"].get(STAGES[k], {}).get("quat_angle_deg", 0) for k in range(9)])
    axes[1].set_ylabel("Rotation error (deg)"); axes[1].set_xticks(range(9), STAGES, rotation=30)
    fig.tight_layout(); fig.savefig(FIG / "offline_error_by_fsm.png", dpi=150); plt.close(fig)

    offsets = sorted(map(int, metrics["chunk_offset_metrics"]))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(offsets, [metrics["chunk_offset_metrics"][str(k)]["xyz_mae_m"] for k in offsets], marker="o")
    axes[0].set_ylabel("XYZ MAE (m)")
    axes[1].plot(offsets, [metrics["chunk_offset_metrics"][str(k)]["quat_angle_deg"] for k in offsets], marker="o")
    axes[1].set_ylabel("Rotation error (deg)"); axes[1].set_xlabel("Action offset within executed 10-step chunk")
    fig.tight_layout(); fig.savefig(FIG / "offline_error_by_chunk_offset.png", dpi=150); plt.close(fig)

    first = grouped[1]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for axis, dim, name in zip(axes, range(3), ("X", "Y", "Z")):
        axis.plot([int(r["policy_step"]) for r in first], [json.loads(r["raw_action"])[dim] for r in first], label="Raw")
        axis.plot([int(r["policy_step"]) for r in first], [json.loads(r["executed_action"])[dim] for r in first], label="Executed")
        axis.set_ylabel(f"{name} (m)"); axis.grid(alpha=.2)
    axes[0].legend(); axes[-1].set_xlabel("Policy step (closed-loop episode 1)")
    fig.tight_layout(); fig.savefig(FIG / "closed_loop_raw_vs_executed.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.step([r["frame"] for r in sample], [1 if r["expert"][7] >= 0 else -1 for r in sample], where="post", label="Expert")
    ax.step([r["frame"] for r in sample], [1 if r["predicted"][7] >= 0 else -1 for r in sample], where="post", label="SmolVLA")
    ax.set(xlabel="Frame (episode 0; three sampled windows)", ylabel="Gripper sign", yticks=[-1, 1]); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "offline_gripper_val_ep0.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for axis, stage in zip(axes, (3, 6)):
        window = transition_groups[(0, stage)]
        axis.step(range(len(window)), [1 if row["expert"][7] >= 0 else -1 for row in window],
                  where="post", label="Expert")
        axis.step(range(len(window)), [1 if row["predicted"][7] >= 0 else -1 for row in window],
                  where="post", label="SmolVLA")
        axis.axvline(10, color="gray", linestyle="--", alpha=.6)
        axis.set_ylabel(f"{STAGES[stage]} sign"); axis.set_yticks([-1, 1]); axis.grid(alpha=.2)
    axes[0].legend(); axes[-1].set_xlabel("Frames around FSM transition (episode 0)")
    fig.tight_layout(); fig.savefig(FIG / "gripper_transition_val_ep0.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    keys = ["workspace_clipped", "speed_limited", "quaternion_adjusted", "action_clipped"]
    x = np.arange(len(keys)); width = .35
    ax.bar(x - width / 2, [metrics["expert_filter_fraction"][k] for k in keys], width, label="Expert")
    ax.bar(x + width / 2, [metrics["model_filter_fraction"][k] for k in keys], width, label="SmolVLA")
    ax.set_xticks(x, ["Workspace", "Speed", "Quat sign", "Any flag"]); ax.set_ylabel("Fraction of steps")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "expert_vs_model_filter_fraction.png", dpi=150); plt.close(fig)
    print(json.dumps(metrics, indent=2, default=encoder))


if __name__ == "__main__":
    main()

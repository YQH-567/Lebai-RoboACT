"""Evaluate an ACT v7 checkpoint on held-out LeRobot episodes without simulation."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame

from act_v7_state import build_state_names


TASK = "Pick up the cube and place it into the tray."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("datasets/lerobot/lebai_lm3_pick_place_v7"))
    args = parser.parse_args()
    manifest = json.loads((args.dataset_root / "split_manifest.json").read_text())
    assert len(manifest["val_source_episode_ids"]) == 40
    assert not set(manifest["train_source_episode_ids"]) & set(manifest["val_source_episode_ids"])
    val_root = args.dataset_root / "val"
    info = json.loads((val_root / "meta/info.json").read_text())
    features = {key: value for key, value in info["features"].items()
                if key.startswith("observation.") or key == "action"}
    names = features["observation.state"]["names"]
    joint_names = [name.removeprefix("qpos_") for name in names[:12]]
    assert names == build_state_names(joint_names)

    tables = [pq.read_table(path, columns=[
        "observation.state", "action", "episode_index"
    ]) for path in sorted((val_root / "data").rglob("*.parquet"))]
    import pyarrow as pa
    data = pa.concat_tables(tables).to_pydict()
    states = np.asarray(data["observation.state"], dtype=np.float32)
    experts = np.asarray(data["action"], dtype=np.float32)
    episode_index = np.asarray(data["episode_index"])
    assert states.shape == (info["total_frames"], 34)
    assert experts.shape == (len(states), 8)
    assert np.isfinite(states).all() and np.isfinite(experts).all()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint.expanduser().resolve()
    model = ACTPolicy.from_pretrained(str(checkpoint), local_files_only=True).to(device).eval()
    assert model.config.chunk_size == 30 and model.config.n_action_steps == 10
    assert model.config.input_features["observation.state"].shape == (34,)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    videos = {
        camera: av.open(
            str(val_root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4")
        ) for camera in ("wrist", "front")
    }
    decoded = {camera: videos[camera].decode(video=0) for camera in videos}
    predictions = []
    previous_episode = None
    try:
        with torch.inference_mode():
            for i, state in enumerate(states):
                episode = int(episode_index[i])
                if episode != previous_episode:
                    model.reset()
                    previous_episode = episode
                values = {name: float(state[j]) for j, name in enumerate(names)}
                for camera in videos:
                    values[camera] = next(decoded[camera]).to_ndarray(format="rgb24")
                frame = build_inference_frame(values, device, features, task=TASK)
                prediction = postprocessor(model.select_action(preprocessor(frame)))
                predictions.append(prediction[0].detach().cpu().numpy())
    finally:
        for video in videos.values():
            video.close()

    predictions = np.asarray(predictions, dtype=np.float64)
    experts = experts.astype(np.float64)
    xyz_mae = np.abs(predictions[:, :3] - experts[:, :3]).mean(axis=0)
    pred_quat = predictions[:, 3:7]
    true_quat = experts[:, 3:7]
    pred_quat /= np.linalg.norm(pred_quat, axis=1, keepdims=True)
    true_quat /= np.linalg.norm(true_quat, axis=1, keepdims=True)
    angles = np.degrees(2 * np.arccos(np.clip(
        np.abs(np.sum(pred_quat * true_quat, axis=1)), 0.0, 1.0
    )))
    gripper_accuracy = np.mean(np.sign(predictions[:, 7]) == np.sign(experts[:, 7]))
    print("Checkpoint:", checkpoint)
    print(f"Validation: {len(set(episode_index))} held-out episodes, {len(states)} frames")
    print("XYZ MAE [m]:", xyz_mae.tolist(), "overall:", float(xyz_mae.mean()))
    print("Quaternion angular error [degree]:", float(angles.mean()))
    print("Gripper accuracy:", float(gripper_accuracy))


if __name__ == "__main__":
    main()

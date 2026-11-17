"""Check ACT v7 episode isolation, state alignment, and unchanged expert actions."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch

import eval_act_isaaclab_v7 as evaluation
from act_v7_state import build_state, build_state_names


def check_live_state_equivalence(f, expected):
    """Exercise the evaluation observation builder with a nonzero env origin."""
    from types import SimpleNamespace

    origin = torch.tensor([[1.25, -0.75, 0.5]], dtype=torch.float32)
    ee = f["observations/ee_pose"][0]
    obj = f["object_position"][0]
    qpos = f["observations/qpos"][0]
    qvel = f["observations/qvel"][0]
    scene = {
        "robot": SimpleNamespace(data=SimpleNamespace(
            joint_pos=torch.tensor(qpos[None]), joint_vel=torch.tensor(qvel[None])
        )),
        "ee_frame": SimpleNamespace(data=SimpleNamespace(
            target_pos_w=torch.tensor(ee[None, None, :3]) + origin[:, None],
            target_quat_w=torch.tensor(ee[None, None, 3:]),
        )),
        "object": SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.tensor(obj[None]) + origin
        )),
    }
    for name, key in (("wrist_camera", "wrist"), ("front_camera", "front")):
        scene[name] = SimpleNamespace(data=SimpleNamespace(
            output={"rgb": torch.tensor(f[f"observations/images/{key}"][0][None])}
        ))
    scene = type("Scene", (dict,), {"env_origins": origin})(scene)
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene))
    actual, wrist, front = evaluation.get_observation(env)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert wrist.shape == front.shape == (180, 320, 3)


def check_split(root, split, ids, raw_dir):
    ds_root = root / split
    info = json.loads((ds_root / "meta/info.json").read_text())
    features = info["features"]
    names = features["observation.state"]["names"]
    assert features["observation.state"]["shape"] == [34]
    assert features["action"]["shape"] == [8]
    assert features["action"]["names"] == evaluation.ACTION_NAMES
    assert names == evaluation.STATE_NAMES == evaluation.DATASET_FEATURES["observation.state"]["names"]
    for camera in ("wrist", "front"):
        key = f"observation.images.{camera}"
        assert features[key]["shape"] == [180, 320, 3]
        assert features[key]["dtype"] == "video"
    assert info["total_episodes"] == len(ids)
    tables = [pq.read_table(path, columns=[
        "observation.state", "action", "episode_index", "frame_index"
    ]) for path in sorted((ds_root / "data").rglob("*.parquet"))]
    if not tables:
        raise RuntimeError(f"Missing {split} Parquet data")
    import pyarrow as pa
    rows = pa.concat_tables(tables).to_pydict()
    states = np.asarray(rows["observation.state"], dtype=np.float32)
    actions = np.asarray(rows["action"], dtype=np.float32)
    ep_index = np.asarray(rows["episode_index"])
    frame_index = np.asarray(rows["frame_index"])
    assert states.shape == (info["total_frames"], 34)
    assert actions.shape == (info["total_frames"], 8)
    assert np.isfinite(states).all() and np.isfinite(actions).all()
    for dataset_ep, source_id in enumerate(ids):
        positions = np.flatnonzero(ep_index == dataset_ep)
        with h5py.File(raw_dir / f"episode_{source_id:06d}.hdf5", "r") as f:
            qpos = f["observations/qpos"][:]
            qvel = f["observations/qvel"][:]
            ee = f["observations/ee_pose"][:]
            obj = f["object_position"][:]
            expert = f["action"][:]
            assert len(positions) == len(expert)
            np.testing.assert_array_equal(frame_index[positions], np.arange(len(expert)))
            expected = np.stack([
                build_state(qpos[i], qvel[i], ee[i], obj[i]) for i in range(len(expert))
            ])
            np.testing.assert_allclose(states[positions], expected, rtol=0, atol=1e-6)
            np.testing.assert_allclose(actions[positions], expert, rtol=0, atol=1e-6)
            if dataset_ep == 0:
                check_live_state_equivalence(f, expected[0])
            if names != build_state_names(json.loads(f.attrs["joint_names"])):
                raise ValueError(f"{split} episode {source_id}: joint names mismatch")
    return len(states)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("datasets/lerobot/lebai_lm3_pick_place_v7"))
    args = parser.parse_args()
    manifest = json.loads((args.dataset_root / "split_manifest.json").read_text())
    train_ids = manifest["train_source_episode_ids"]
    val_ids = manifest["val_source_episode_ids"]
    assert len(train_ids) == 160 and len(val_ids) == 40
    assert len(set(train_ids)) == 160 and len(set(val_ids)) == 40
    assert not set(train_ids).intersection(val_ids)
    assert manifest["train_dataset_episode_to_source"] == train_ids
    assert manifest["val_dataset_episode_to_source"] == val_ids
    raw_dir = Path(manifest["source_dir"])
    source_ids = sorted(int(path.stem.removeprefix("episode_"))
                        for path in raw_dir.glob("episode_*.hdf5"))
    expected = np.random.default_rng(manifest["seed"]).permutation(source_ids).tolist()
    assert train_ids + val_ids == expected
    train_frames = check_split(args.dataset_root, "train", train_ids, raw_dir)
    val_frames = check_split(args.dataset_root, "val", val_ids, raw_dir)
    assert train_frames == manifest["train_frames"]
    assert val_frames == manifest["val_frames"]
    print(f"PASS: 160/40 disjoint episodes; {train_frames}/{val_frames} frames")
    print("PASS: all 34D states finite and equal to HDF5; live-state builder matches")
    print("PASS: all 8D actions equal to HDF5; image features unchanged")


if __name__ == "__main__":
    main()

"""Shared ACT v7 state definition for HDF5 conversion and Isaac Lab evaluation."""

import numpy as np


OBJECT_REL_NAMES = ["object_rel_x", "object_rel_y", "object_rel_z"]


def build_state_names(joint_names):
    if len(joint_names) != 12:
        raise ValueError(f"Expected 12 joints, got {len(joint_names)}")
    return (
        [f"qpos_{name}" for name in joint_names]
        + [f"qvel_{name}" for name in joint_names]
        + ["ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz"]
        + OBJECT_REL_NAMES
    )


def build_state(qpos, qvel, ee_pose, object_position):
    """Build [qpos(12), qvel(12), EE pose(7), object-EE XYZ(3)]."""
    qpos = np.asarray(qpos, dtype=np.float32)
    qvel = np.asarray(qvel, dtype=np.float32)
    ee_pose = np.asarray(ee_pose, dtype=np.float32)
    object_position = np.asarray(object_position, dtype=np.float32)
    if (qpos.shape, qvel.shape, ee_pose.shape, object_position.shape) != (
        (12,), (12,), (7,), (3,)
    ):
        raise ValueError("Expected qpos(12), qvel(12), ee_pose(7), object_position(3)")
    state = np.concatenate(
        [qpos, qvel, ee_pose, object_position - ee_pose[:3]]
    ).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("ACT v7 observation.state contains NaN or Inf")
    return state

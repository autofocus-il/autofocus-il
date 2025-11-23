import numpy as np
from typing import Dict, List
from pyquaternion import Quaternion


def stack_obs(obs: List[Dict]):
    """
    Stack a sequence of observations of length T into the first dimension of a batch.
    - Input: obs = [{"image":..., "proprio":...}, {..}, ...]
    - Output: {"image": np.stack([...], axis=0), "proprio": np.stack([...], axis=0), ...}
    """
    if not obs:
        return {}
    keys = list(obs[0].keys())
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        vals = [o[k] for o in obs]
        out[k] = np.stack(vals, axis=0)
    return out


def state_to_eep(xyz_coor, zangle: float):
    """
    Implement the state to eep function.
    Refered to `bridge_data_robot`'s `widowx_controller/widowx_controller.py`
    return a 4x4 matrix
    """
    assert len(xyz_coor) == 3
    DEFAULT_ROTATION = np.array([
        [0, 0, 1.0],
        [0, 1.0, 0],
        [-1.0, 0, 0],
    ])
    new_pose = np.eye(4)
    new_pose[:3, -1] = xyz_coor
    new_quat = Quaternion(axis=np.array([0.0, 0.0, 1.0]), angle=zangle) * Quaternion(matrix=DEFAULT_ROTATION)
    new_pose[:3, :3] = new_quat.rotation_matrix
    return new_pose


def mat_to_xyzrpy(mat: np.ndarray):
    """return a 6-dim vector with xyz and rpy"""
    assert mat.shape == (4, 4), "mat must be a 4x4 matrix"
    xyz = mat[:3, -1]
    quat = Quaternion(matrix=mat[:3, :3])
    yaw, pitch, roll = quat.yaw_pitch_roll
    return np.concatenate([xyz, [roll, pitch, yaw]])

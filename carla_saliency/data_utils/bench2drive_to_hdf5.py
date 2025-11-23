#!/usr/bin/env python3
"""
Bench2Drive -> Robomimic HDF5 converter

Converts the Bench2Drive-style directory of episodes into a single HDF5
file readable by robomimic.utils.dataset.SequenceDataset.

Usage:
  python -m carla_saliency.data_utils.bench2drive_to_hdf5 --config carla_saliency/configs/bench2drive_to_hdf5.yaml

Expected source directory layout (example):
  <root>/bench2drive220/route_2416/seed_200/
      observations.pt     # [T, H, W, 3] uint8 (may be npz or ndarray/tensor)
      actions.pt          # [T, A] (may be npz or dict with 'actions')
      gaze.pt             # gaze-like annotations (variable formats)
      gaze_pseudo.pt      # same as above
      filter_dynamic.pt   # same as above
      non_filter.pt       # same as above
      stats.json          # (optional)

Output HDF5 layout (robomimic-compatible):
  data/demo_0/attrs[num_samples]
  data/demo_0/obs/image              [T, H, W, 3] uint8
  data/demo_0/obs/gaze_coords        [T, max_points*2] float32, coords in [0,1] (legacy alias)
  data/demo_0/obs/gaze_coords_gaze             [T, max_points*2] float32
  data/demo_0/obs/gaze_coords_gaze_pseudo      [T, max_points*2] float32
  data/demo_0/obs/gaze_coords_filter_dynamic   [T, max_points*2] float32
  data/demo_0/obs/gaze_coords_non_filter       [T, max_points*2] float32
  data/demo_0/next_obs/image         [T, H, W, 3] uint8 (shifted with last repeated)
  data/demo_0/next_obs/gaze_coords   [T, max_points*2] float32 (legacy alias)
  data/demo_0/next_obs/gaze_coords_gaze             [T, max_points*2] float32
  data/demo_0/next_obs/gaze_coords_gaze_pseudo      [T, max_points*2] float32
  data/demo_0/next_obs/gaze_coords_filter_dynamic   [T, max_points*2] float32
  data/demo_0/next_obs/gaze_coords_non_filter       [T, max_points*2] float32
  data/demo_0/actions                [T, A] float32
  data/demo_0/rewards                [T, 1] float32 (zeros by default)
  data/demo_0/dones                  [T, 1] float32 (last = 1.0)

Config (YAML) keys (minimal):
  dataset_root: "/path/to/dataset/bench2drive220"
  output_hdf5:  "/path/to/dataset/bench2drive220_robomimic.hdf5"
  include_gaze: true
  include_gaze_pseudo: true
  include_filter_dynamic: true
  include_non_filter: true
  max_gaze_points: 5
  action_dim: 7
  compression: "lzf"          # or null, or "gzip"
  chunk_len: 256               # chunking on time dimension for writing
  limit_episodes: null         # or an int
  include_routes: []           # optional whitelist of route_* directories
  include_seeds: []            # optional whitelist of seed_* directories
  skip_on_error: true

Note: this script does not require robomimic as a dependency.
"""

from __future__ import annotations

import os
import argparse
import time
from typing import List, Optional, Tuple

import numpy as np
import h5py
from tqdm import tqdm

# torch is optional but very helpful for reading .pt files robustly
try:
    import torch
except Exception:  # pragma: no cover - fallback without torch
    torch = None

# YAML is optional; if unavailable, config can be provided via CLI args
try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

"""
Note: Simplified script
- Uses only `convert_to_robomimic_format` for conversion logic (as requested).
- Keeps the HDF5 layout and the YAML configuration flow (dataset root, output
  path, compression on/off via `compression` key, chunk length, and route/seed
  filters). The include_* gaze toggles are ignored by the conversion function,
  but layout still stores all found variants and a legacy alias.
"""


# -----------------------------------------------------------------------------
# Helpers: file loading & preprocessing
# -----------------------------------------------------------------------------


def _log(msg: str, verbose: bool = True):
    if verbose:
        print(msg)


def load_pt_any(path: str):
    """Robust loader for .pt files that may contain numpy arrays or tensors.

    Returns a numpy array or Python object.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def pick_from_npz(
    npz: "np.lib.npyio.NpzFile",
    preferred_keys=(
        "observations",
        "images",
        "rgb",
        "frames",
        "values",
        "arr_0",
        "arr0",
        "actions",
        "acts",
    ),
):
    """Pick a sensible array from an npz-like object.

    Preference order: preferred_keys -> any array-like with ndim in {3,4} or object arrays -> first key.
    """
    keys = list(getattr(npz, "files", []) or [])
    # direct preferred keys
    for k in preferred_keys:
        if k in npz:
            return npz[k]
        if k in keys:
            return npz[k]
    # search for array-like shapes likely to be images
    for k in keys:
        v = npz[k]
        try:
            if isinstance(v, np.ndarray):
                if v.dtype == object or v.ndim in (3, 4) or v.ndim == 0:
                    return v
        except Exception:
            continue
    # fallback: first
    if keys:
        return npz[keys[0]]
    # Some np.load returns array directly; return as-is
    return np.asarray(npz)


def unwrap_0d(x):
    """Recursively unwrap 0-d numpy arrays that wrap a Python object (e.g., ndarray inside)."""
    while isinstance(x, np.ndarray) and x.ndim == 0:
        try:
            x = x.item()
        except Exception:
            break
    return x


def maybe_unpickle_bytes(x):
    """If x is pickled bytes (or 0-d array containing bytes), unpickle it."""
    import pickle

    # numpy 0-d wrappers
    if isinstance(x, np.ndarray) and x.ndim == 0:
        try:
            x = x.item()
        except Exception:
            return x
    # raw bytes or bytearray
    if isinstance(x, (bytes, bytearray)):
        # common pickle header starts with 0x80 0x02
        try:
            return pickle.loads(bytes(x))
        except Exception:
            return x
    # numpy bytes array
    if isinstance(x, np.ndarray) and x.dtype.kind in ("S", "V"):
        try:
            return maybe_unpickle_bytes(x.tobytes())
        except Exception:
            return x
    return x


def load_observations(obs_path: str) -> np.ndarray:
    """Load observations via torch only, then coerce to [T,H,W,3] uint8.

    This assumes `torch.load(obs_path, weights_only=False)` returns an ndarray
    with shape like [T,H,W,3] (or a torch tensor convertible to that).
    """
    if torch is None:
        raise ImportError("PyTorch is required to load observations")
    obj = torch.load(obs_path, map_location="cpu", weights_only=False)
    # Convert to numpy if tensor
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
    else:
        arr = obj
    arr = unwrap_0d(arr)
    return ensure_image_uint8(arr)


def ensure_image_uint8(arr_in) -> np.ndarray:
    """Robustly coerce observations into uint8 [T, H, W, 3].

    Accepts a variety of layouts and container types:
      - np.ndarray with shape [T,H,W,C], [T,C,H,W], [H,W,C], [H,W]
      - lists / tuples of frames -> stacks along time dim
      - object arrays or zero-d arrays (unwrap .item())
    """

    def to_ndarray(x):
        # Unwrap 0-d arrays
        if isinstance(x, np.ndarray) and x.ndim == 0:
            try:
                x = x.item()
            except Exception:
                pass
        # If it's an NPZ container, pick a likely array
        try:
            if isinstance(x, np.lib.npyio.NpzFile):
                x = pick_from_npz(x)
        except Exception:
            pass
        # Convert lists/tuples of frames
        if isinstance(x, (list, tuple)):
            try:
                return np.stack([to_ndarray(f) for f in x], axis=0)
            except Exception:
                x = np.array(x, dtype=object)
        return np.asarray(x)

    arr = to_ndarray(arr_in)

    # If object array of frames, try to stack
    if arr.dtype == object:
        try:
            frames = [np.asarray(f) for f in arr.tolist()]
            arr = np.stack(frames, axis=0)
        except Exception:
            pass

    # Handle common shapes
    if arr.ndim == 2:
        # [H, W] -> [1, H, W, 1]
        arr = arr[None, ..., None]
    elif arr.ndim == 3:
        # Could be [H,W,C] or [C,H,W]
        if arr.shape[-1] in (1, 3):
            arr = arr[None, ...]  # [1,H,W,C]
        elif arr.shape[0] in (1, 3):
            # [C,H,W] -> [1,H,W,C]
            arr = np.moveaxis(arr, 0, -1)[None, ...]
        else:
            raise ValueError(
                f"observations.pt must be [T,H,W,1/3], got shape {arr.shape}"
            )
    elif arr.ndim == 4:
        # [T,H,W,C] or [T,C,H,W]
        if arr.shape[-1] in (1, 3):
            pass
        elif arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            # [T,C,H,W] -> [T,H,W,C]
            arr = np.moveaxis(arr, 1, -1)
        else:
            raise ValueError(
                f"observations.pt must be [T,H,W,1/3], got shape {arr.shape}"
            )
    else:
        raise ValueError(f"observations.pt must be [T,H,W,1/3], got shape {arr.shape}")

    # Ensure channel last and 3 channels
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    # Dtype to uint8
    if arr.dtype != np.uint8:
        # If likely normalized floats [0,1], scale to [0,255]
        a = arr.astype(np.float32)
        if np.nanmax(a) <= 1.0 + 1e-6:
            a = a * 255.0
        arr = np.clip(a, 0, 255).astype(np.uint8)
    return arr


def as_float32(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).astype(np.float32, copy=False)


def process_gaze_array(
    gaze: np.ndarray,
    T: int,
    H: int,
    W: int,
    max_points: int,
) -> np.ndarray:
    """Normalize and pad/clip gaze coords.

    Input forms supported:
      - [T, P, 2]
      - [T, P*2]
      - [T, 2]  (P=1)
    Values may be pixel coords or normalized in [0,1]. If any value > 1, assume pixels
    and normalize by width (x) / height (y).

    Returns [T, max_points*2] float32 with missing entries filled with -1.
    """
    g = np.asarray(gaze)
    if g.ndim == 2 and g.shape[-1] == 2:
        g = g[:, None, :]  # [T, 1, 2]
    elif g.ndim == 2:
        # [T, P*2] -> [T, P, 2]
        P = g.shape[-1] // 2
        g = g.reshape(g.shape[0], P, 2)
    elif g.ndim == 3 and g.shape[-1] == 2:
        pass  # already [T, P, 2]
    else:
        raise ValueError(f"Unsupported gaze shape: {g.shape}")

    if g.shape[0] != T:
        # align time dimension conservatively by trimming/padding
        T_eff = min(T, g.shape[0])
        g = g[:T_eff]
        if T_eff < T:
            pad = np.full((T - T_eff, g.shape[1], 2), -1.0, dtype=np.float32)
            g = np.concatenate([g, pad], axis=0)

    # Normalize if values look like pixels
    if np.nanmax(np.abs(g)) > 1.0:
        # (x, y), width = W, height = H
        g = g.astype(np.float32)
        x = g[..., 0] / max(W - 1, 1)
        y = g[..., 1] / max(H - 1, 1)
        g = np.stack([x, y], axis=-1)
    else:
        g = g.astype(np.float32)

    # Pad or clip to max_points
    P_in = g.shape[1]
    if P_in < max_points:
        pad = np.full((g.shape[0], max_points - P_in, 2), -1.0, dtype=np.float32)
        g = np.concatenate([g, pad], axis=1)
    elif P_in > max_points:
        g = g[:, :max_points, :]

    g = g.reshape(g.shape[0], max_points * 2)
    return g.astype(np.float32)


def _to_points_2d_array(frame_obj) -> np.ndarray:
    """Coerce a single-frame gaze-like item into [P,2] array.
    Accepts list/tuple/ndarray with shape variations; computes centers for 4-value boxes.
    Returns empty array when cannot parse.
    """
    try:
        # Direct ndarray
        if isinstance(frame_obj, np.ndarray):
            arr = frame_obj
        else:
            arr = np.asarray(frame_obj, dtype=np.float32)
        if arr.ndim == 1:
            # Accept 1D vectors of length >= 2
            if arr.size >= 2:
                if arr.size % 2 == 0:
                    return arr.reshape(-1, 2).astype(np.float32)
                # Odd length (e.g., (x,y,z)): take first two as (x,y)
                return arr[:2].reshape(1, 2).astype(np.float32)
            return np.zeros((0, 2), dtype=np.float32)
        if arr.ndim >= 2:
            # If it looks like [P,4] (x1,y1,x2,y2) -> center
            if arr.shape[-1] == 4:
                x1, y1, x2, y2 = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                pts = np.stack([cx, cy], axis=-1)
                return pts.reshape(-1, 2).astype(np.float32)
            # If last dim >=2, take first two as (x,y)
            if arr.shape[-1] >= 2:
                return (
                    arr.reshape(-1, arr.shape[-1])[..., :2]
                    .reshape(-1, 2)
                    .astype(np.float32)
                )
        return np.zeros((0, 2), dtype=np.float32)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)


def process_gaze_like(
    raw_obj,
    T: int,
    H: int,
    W: int,
    max_points: int,
) -> np.ndarray:
    """Handle both ndarray-like and list-of-list gaze formats.

    Returns [T, max_points*2] float32 with coords normalized to [0,1] and -1 padding.
    """
    # If ndarray-ish, reuse existing path
    if isinstance(raw_obj, np.ndarray):
        return process_gaze_array(raw_obj, T=T, H=H, W=W, max_points=max_points)
    # torch tensor path (if import happened earlier)
    if torch is not None and isinstance(raw_obj, torch.Tensor):
        return process_gaze_array(
            raw_obj.detach().cpu().numpy(), T=T, H=H, W=W, max_points=max_points
        )

    # Otherwise treat as sequence of frames (lists/tuples)
    if isinstance(raw_obj, (list, tuple)):
        frames = list(raw_obj)
        # Build per-frame points arrays
        pts_per_frame = []
        for f in frames:
            pts = _to_points_2d_array(f)
            pts_per_frame.append(pts)
        # Normalize and pad/clip per frame
        out = np.full((max(T, len(frames)), max_points, 2), -1.0, dtype=np.float32)
        for t in range(min(T, len(frames))):
            pts = pts_per_frame[t]
            if pts.size == 0:
                continue
            # Normalize if looks like pixels
            if np.nanmax(np.abs(pts)) > 1.0:
                xs = pts[:, 0] / max(W - 1, 1)
                ys = pts[:, 1] / max(H - 1, 1)
                pts = np.stack([xs, ys], axis=-1).astype(np.float32)
            else:
                pts = pts.astype(np.float32)
            if pts.shape[0] > 0:
                pts = pts[:max_points]
                out[t, : pts.shape[0], :] = pts
        return out[:T].reshape(T, max_points * 2).astype(np.float32)

    # Unknown type -> fill with -1
    return np.full((T, max_points * 2), -1.0, dtype=np.float32)


def shift_next(arr: np.ndarray) -> np.ndarray:
    """Shift sequence by -1 with last frame repeated: [x1..xT] -> [x2..xT xT]."""
    T = arr.shape[0]
    if T == 0:
        return arr
    return np.concatenate([arr[1:], arr[-1:]], axis=0)


# -----------------------------------------------------------------------------
# API-compatible conversion (as requested): convert_to_robomimic_format
# -----------------------------------------------------------------------------


def convert_to_robomimic_format(
    episodes: List[Tuple[int, int]],
    datapath: str,
    output_path: str = "/path/to/dataset/bench2drive220_robomimic.hdf5",
    compress: bool = True,
    chunk_size: int = 64,
    pseudo_gaze: bool = True,
    include_gaze: bool = True,
    include_gaze_pseudo: bool = True,
    include_filter_dynamic: bool = True,
    include_non_filter: bool = True,
    max_gaze_points: int = 5,
):
    """
    Convert Bench2Drive dataset to a robomimic-compatible HDF5 file.

    This function mirrors the requested signature and high-level behavior while
    preserving the existing HDF5 layout used in this repository, namely:
      - Stores images at data/demo_X/obs/image and next_obs/image
      - Stores multiple gaze variants at data/demo_X/obs/gaze_coords_<variant>
        and next_obs/gaze_coords_<variant>
      - Provides a legacy alias 'gaze_coords' (and in next_obs) that points to a
        chosen variant (gaze_pseudo when available/selected, else gaze, etc.)
      - Stores actions, rewards, and dones at the demo level

    Parameters
    - episodes: list of (route_id, seed) integer pairs
    - datapath: root directory that contains route_{id}/seed_{seed} subfolders
    - output_path: output HDF5 file path
    - compress: whether to apply lightweight compression (lzf)
    - chunk_size: HDF5 chunk length along the time dimension
    - pseudo_gaze: whether the legacy alias 'gaze_coords' should prefer
      gaze_pseudo over other variants if present
    """

    print(f"Converting dataset to Robomimic HDF5 format: {output_path}")
    print(f"Processing {len(episodes)} episodes...")

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    with h5py.File(output_path, "w", libver="latest") as f:
        # Global attributes
        f.attrs["total_episodes"] = len(episodes)
        f.attrs["pseudo_gaze"] = bool(pseudo_gaze)
        f.attrs["creation_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        f.attrs["env_name"] = "CARLA_Bench2Drive"

        g_data = f.create_group("data")

        valid_demos: List[str] = []
        for ep_idx, (route_id, seed) in enumerate(
            tqdm(episodes, desc="Converting episodes")
        ):
            base_path = os.path.join(datapath, f"route_{route_id}", f"seed_{seed}")

            obs_path = os.path.join(base_path, "observations.pt")
            if not os.path.exists(obs_path):
                print(f"Warning: Episode not found: route_{route_id}/seed_{seed}")
                continue

            # demo_N format as required by robomimic
            demo_name = f"demo_{len(valid_demos)}"
            demo_group = g_data.create_group(demo_name)

            # Load and normalize observations to [T,H,W,3] uint8
            if torch is None:
                raise ImportError("PyTorch is required to load observations")
            obs_obj = torch.load(obs_path, map_location="cpu", weights_only=False)
            if isinstance(obs_obj, torch.Tensor):
                obs_arr = obs_obj.detach().cpu().numpy()
            else:
                obs_arr = obs_obj
            obs = ensure_image_uint8(obs_arr)

            T = int(obs.shape[0])

            # Episode metadata
            demo_group.attrs["num_samples"] = T
            demo_group.attrs["route_id"] = int(route_id)
            demo_group.attrs["seed"] = int(seed)
            demo_group.attrs["episode_idx"] = int(ep_idx)

            obs_group = demo_group.create_group("obs")
            next_obs_group = demo_group.create_group("next_obs")

            # Datasets: images
            chunks = (
                (min(T, int(chunk_size)),) + tuple(obs.shape[1:]) if compress else None
            )
            obs_group.create_dataset(
                "image",
                data=obs,
                chunks=chunks,
                compression=("lzf" if compress else None),
            )

            next_images = shift_next(obs)
            next_obs_group.create_dataset(
                "image",
                data=next_images,
                chunks=chunks,
                compression=("lzf" if compress else None),
            )

            # Actions
            actions_path = os.path.join(base_path, "actions.pt")
            if os.path.exists(actions_path):
                actions_data = load_pt_any(actions_path)
                actions = actions_data["actions"]
                demo_group.create_dataset(
                    "actions",
                    data=actions,
                    chunks=True if compress else None,
                    compression=("lzf" if compress else None),
                )

            # Gaze variants: load all available, then choose alias
            # Keep repo layout: write per-variant under gaze_coords_<variant>
            H, W = int(obs.shape[1]), int(obs.shape[2])
            max_points = int(max_gaze_points)

            variant_files = {
                "gaze": os.path.join(base_path, "gaze.pt"),
                "gaze_pseudo": os.path.join(base_path, "gaze_pseudo.pt"),
                "filter_dynamic": os.path.join(base_path, "filter_dynamic.pt"),
                "non_filter": os.path.join(base_path, "non_filter.pt"),
            }
            variant_include = {
                "gaze": bool(include_gaze),
                "gaze_pseudo": bool(include_gaze_pseudo),
                "filter_dynamic": bool(include_filter_dynamic),
                "non_filter": bool(include_non_filter),
            }
            processed_variants: dict[str, np.ndarray] = {}
            for vkey, vpath in variant_files.items():
                if not variant_include.get(vkey, True):
                    continue  # disabled by include_* toggle
                if not os.path.exists(vpath):
                    continue
                raw = load_pt_any(vpath)
                arr = process_gaze_like(raw, T=T, H=H, W=W, max_points=max_points)
                processed_variants[vkey] = arr

            alias_only_placeholder: Optional[np.ndarray] = None
            if not processed_variants:
                # No enabled+present variants. Still create legacy alias to keep layout.
                alias_only_placeholder = np.full(
                    (T, max_points * 2), -1.0, dtype=np.float32
                )

            # Write per-variant datasets
            for vkey, varr in processed_variants.items():
                obs_group.create_dataset(
                    f"gaze_coords_{vkey}",
                    data=varr.astype(np.float32),
                    chunks=True if compress else None,
                    compression=("lzf" if compress else None),
                )
                next_obs_group.create_dataset(
                    f"gaze_coords_{vkey}",
                    data=shift_next(varr).astype(np.float32),
                    chunks=True if compress else None,
                    compression=("lzf" if compress else None),
                )

            # Legacy alias: prefer pseudo gaze when requested and present
            prefer_order_base = (
                ["gaze_pseudo", "gaze", "filter_dynamic", "non_filter"]
                if pseudo_gaze
                else ["gaze", "gaze_pseudo", "filter_dynamic", "non_filter"]
            )
            # Filter prefer order by include_* toggles
            prefer_order = [
                k for k in prefer_order_base if variant_include.get(k, True)
            ]
            if processed_variants:
                alias_key = next(
                    (k for k in prefer_order if k in processed_variants),
                    next(iter(processed_variants.keys())),
                )
                alias_arr = processed_variants[alias_key]
            else:
                alias_arr = alias_only_placeholder
            obs_group.create_dataset(
                "gaze_coords",
                data=alias_arr.astype(np.float32),
                chunks=True if compress else None,
                compression=("lzf" if compress else None),
            )
            next_obs_group.create_dataset(
                "gaze_coords",
                data=shift_next(alias_arr).astype(np.float32),
                chunks=True if compress else None,
                compression=("lzf" if compress else None),
            )

            # Rewards and dones
            rewards = np.zeros((T, 1), dtype=np.float32)
            demo_group.create_dataset(
                "rewards",
                data=rewards,
                chunks=True if compress else None,
                compression=("lzf" if compress else None),
            )

            dones = np.zeros((T, 1), dtype=np.float32)
            dones[-1, 0] = 1.0
            demo_group.create_dataset(
                "dones",
                data=dones,
                chunks=True if compress else None,
                compression=("lzf" if compress else None),
            )

            valid_demos.append(demo_name)

        # Summary attrs and flush
        f.attrs["num_demos"] = len(valid_demos)
        f.flush()

    print(f"HDF5 conversion complete: {output_path}")
    print(f"Total valid demos: {len(valid_demos)}")

    # Simple file structure verification
    with h5py.File(output_path, "r") as f:
        print(f"\nFile structure verification:")
        print(f" Root keys: {list(f.keys())}")
        if "data" in f:
            demos = list(f["data"].keys())
            print(f" Number of demos: {len(demos)}")
            if demos:
                demo = demos[0]
                print(f" First demo: {demo}")
                print(f" Keys: {list(f['data'][demo].keys())}")
                print(f" Obs keys: {list(f['data'][demo]['obs'].keys())}")
                print(f" Num samples: {f['data'][demo].attrs['num_samples']}")
        try:
            size_gb = os.path.getsize(output_path) / (1024**3)
            print(f" File size: {size_gb:.2f} GB")
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Minimal discovery utils to feed convert_to_robomimic_format
# -----------------------------------------------------------------------------


def discover_episodes_simple(
    dataset_root: str,
    include_routes: Optional[List[str]] = None,
    include_seeds: Optional[List[str]] = None,
    limit_episodes: Optional[int] = None,
) -> List[Tuple[int, int]]:
    include_routes = include_routes or []
    include_seeds = include_seeds or []
    routes = [
        d
        for d in sorted(os.listdir(dataset_root))
        if d.startswith("route_") and os.path.isdir(os.path.join(dataset_root, d))
    ]
    if include_routes:
        routes = [r for r in routes if r in include_routes]
    pairs: List[Tuple[int, int]] = []
    for rd in routes:
        rdir = os.path.join(dataset_root, rd)
        seeds = [
            d
            for d in sorted(os.listdir(rdir))
            if d.startswith("seed_") and os.path.isdir(os.path.join(rdir, d))
        ]
        if include_seeds:
            seeds = [s for s in seeds if s in include_seeds]
        try:
            rid = int(rd.split("_")[-1])
        except Exception:
            continue
        for sd in seeds:
            try:
                sid = int(sd.split("_")[-1])
            except Exception:
                continue
            pairs.append((rid, sid))
    if limit_episodes is not None:
        pairs = pairs[: int(limit_episodes)]
    return pairs


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def load_yaml_config(path: Optional[str]) -> Optional[dict]:
    if path is None:
        return None
    if yaml is None:
        raise ImportError(
            "PyYAML is not installed. Please install pyyaml or pass CLI args instead."
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_config_and_args(args: argparse.Namespace) -> dict:
    base = load_yaml_config(args.config) or {}

    # CLI can override a subset of keys
    def pick(key, default=None):
        return (
            getattr(args, key)
            if getattr(args, key) is not None
            else base.get(key, default)
        )

    cfg = {
        "dataset_root": pick(
            "dataset_root", "/path/to/dataset/bench2drive220"
        ),
        "output_hdf5": pick(
            "output_hdf5", "/path/to/dataset/bench2drive220_robomimic.hdf5"
        ),
        "compression": pick("compression", "lzf"),
        "chunk_len": int(pick("chunk_len", 256)),
        "max_gaze_points": int(pick("max_gaze_points", 5)),
        "limit_episodes": pick("limit_episodes", None),
        "include_routes": pick("include_routes", []) or [],
        "include_seeds": pick("include_seeds", []) or [],
        # Include toggles for writing variants
        "include_gaze": bool(pick("include_gaze", True)),
        "include_gaze_pseudo": bool(pick("include_gaze_pseudo", True)),
        "include_filter_dynamic": bool(pick("include_filter_dynamic", True)),
        "include_non_filter": bool(pick("include_non_filter", True)),
        # Alias preference (defaults to follow include_gaze_pseudo)
        "pseudo_gaze": bool(pick("include_gaze_pseudo", True)),
        "verbose": bool(pick("verbose", True)),
    }
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bench2Drive -> Robomimic HDF5 converter (simple)"
    )
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument(
        "--dataset-root", type=str, default=None, help="Override dataset root path"
    )
    p.add_argument(
        "--output-hdf5", type=str, default=None, help="Override output HDF5 path"
    )
    p.add_argument(
        "--compression",
        type=str,
        default=None,
        choices=[None, "lzf", "gzip"],
        nargs="?",
    )
    p.add_argument("--chunk-len", type=int, default=None)
    p.add_argument("--max-gaze-points", type=int, default=None)
    p.add_argument("--limit-episodes", type=int, default=None)
    p.add_argument("--include-routes", type=str, nargs="*", default=None)
    p.add_argument("--include-seeds", type=str, nargs="*", default=None)
    p.add_argument(
        "--include-gaze",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=None,
    )
    p.add_argument(
        "--include-gaze-pseudo",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=None,
    )
    p.add_argument(
        "--include-filter-dynamic",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=None,
    )
    p.add_argument(
        "--include-non-filter",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=None,
    )
    p.add_argument(
        "--verbose", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=None
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = parse_config_and_args(args)

    dataset_root = cfg["dataset_root"]
    output_hdf5 = cfg["output_hdf5"]
    compress = cfg["compression"] is not None  # lzf by default in function
    chunk_len = cfg["chunk_len"]
    pseudo_gaze = cfg["pseudo_gaze"]
    max_gaze_points = cfg["max_gaze_points"]
    include_gaze = cfg["include_gaze"]
    include_gaze_pseudo = cfg["include_gaze_pseudo"]
    include_filter_dynamic = cfg["include_filter_dynamic"]
    include_non_filter = cfg["include_non_filter"]

    episodes = discover_episodes_simple(
        dataset_root=dataset_root,
        include_routes=cfg.get("include_routes") or [],
        include_seeds=cfg.get("include_seeds") or [],
        limit_episodes=cfg.get("limit_episodes"),
    )
    if len(episodes) == 0:
        raise RuntimeError(f"No episodes found under {dataset_root}")

    print("\n==== Bench2Drive -> Robomimic HDF5 Converter (simple) ====")
    print(f"Source : {dataset_root}")
    print(f"Output : {output_hdf5}")
    print(f"Episodes: {len(episodes)}")
    print(f"Compress: {compress} (uses lzf)")
    print(f"Chunk len: {chunk_len}")
    print(f"Max gaze points: {max_gaze_points}")
    print(f"Alias prefers pseudo gaze: {pseudo_gaze}")
    print(
        f"Include variants -> gaze:{include_gaze} pseudo:{include_gaze_pseudo} filter_dynamic:{include_filter_dynamic} non_filter:{include_non_filter}"
    )
    print("========================================================\n")

    convert_to_robomimic_format(
        episodes=episodes,
        datapath=dataset_root,
        output_path=output_hdf5,
        compress=compress,
        chunk_size=chunk_len,
        pseudo_gaze=pseudo_gaze,
        include_gaze=include_gaze,
        include_gaze_pseudo=include_gaze_pseudo,
        include_filter_dynamic=include_filter_dynamic,
        include_non_filter=include_non_filter,
        max_gaze_points=max_gaze_points,
    )


if __name__ == "__main__":
    main()

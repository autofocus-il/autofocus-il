"""
BDV2 Dataset Analysis

This script analyzes the BridgeDataset v2 style dataset layout located at
`/path/to/dataset/bdv2` (configurable via CLI) and prints a
concise summary of its structure and the per-trajectory signal dimensions
such as observations and actions.

Directory structure example (discovered on disk):
  {dataset_root}/
    {task_name}/
      {timestamp}/
        collection_metadata.json
        config.json
        diagnostics.png
        raw/
          traj_group{N}/
            traj{K}/
              images0/ im_0.jpg ...
              images1/
              images2/
              obs_dict.pkl
              policy_out.pkl
              agent_data.pkl
              lang.txt

Key outputs:
- Counts of tasks, sessions, traj groups, and trajectories.
- For each sampled trajectory: image counts per camera folder, obs_dict
  key shapes/dtypes, action vector dimension and step count.
- Mismatch warnings between image frames, observation steps, and action steps.

Notes:
- Some pickles (e.g., policy_out.pkl, agent_data.pkl) may contain ROS types.
  We use a SafeUnpickler that substitutes unknown classes with dummies so we
  can still introspect basic Python containers to recover action shapes.

Usage:
  python -m refactor.bdv2_data_analysis \
      --dataset-root /path/to/dataset/bdv2 \
      --task open_microwave \
      --limit 3 \
      --save-json refactor/bdv2_inspect/analysis_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# -------------------------------
# Safe pickle loading for ROS-msg
# -------------------------------
class _DummyClass:  # pragma: no cover - simple placeholder
    def __init__(self, *a, **k):
        pass


class SafeUnpickler(pickle.Unpickler):  # pragma: no cover - IO helper
    """Unpickler that returns dummy classes when modules aren't importable.

    This allows us to load pickles containing ROS types sufficiently to
    inspect surrounding Python container structures (lists/dicts/ndarrays).
    """

    def find_class(self, module, name):
        try:
            __import__(module)
            mod = sys.modules[module]
            return getattr(mod, name)
        except Exception:
            # Fall back to a simple dummy class; sufficient for container shape inspection
            return type(name, (_DummyClass,), {})


def safe_pickle_load(path: Path) -> Any:  # pragma: no cover - IO helper
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()


# -------------------------
# Data structures & helpers
# -------------------------
@dataclass
class TrajImageInfo:
    camera_counts: Dict[str, int]
    cameras_found: List[str]


@dataclass
class TrajObsInfo:
    step_count: Optional[int]
    key_shapes: Dict[str, Tuple[int, ...]]
    key_dtypes: Dict[str, str]


@dataclass
class TrajActionInfo:
    step_count: Optional[int]
    action_dim: Optional[int]
    policy_type: Optional[str]
    always_zero_dims: Optional[List[int]] = None
    min_per_dim: Optional[List[float]] = None
    max_per_dim: Optional[List[float]] = None
    curve_image_path: Optional[str] = None


@dataclass
class TrajAnalysis:
    task: str
    timestamp: str
    traj_group: str
    traj_name: str
    traj_path: str
    images: TrajImageInfo
    observations: TrajObsInfo
    actions: TrajActionInfo
    warnings: List[str]


def _numeric_sort_key(name: str) -> Tuple[str, int]:
    # Extract trailing integer from filenames like im_10.jpg for correct order
    base = os.path.splitext(os.path.basename(name))[0]
    try:
        idx = int(base.split("_")[-1])
    except Exception:
        idx = -1
    return (base, idx)


def analyze_images(traj_dir: Path) -> TrajImageInfo:
    camera_counts: Dict[str, int] = {}
    cameras_found: List[str] = []
    for cam_idx in range(4):  # probe common names images0..images3
        cam_dir = traj_dir / f"images{cam_idx}"
        if cam_dir.exists() and cam_dir.is_dir():
            cameras_found.append(f"images{cam_idx}")
            files = [p for p in cam_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            files.sort(key=lambda p: _numeric_sort_key(p.name))
            camera_counts[f"images{cam_idx}"] = len(files)
    return TrajImageInfo(camera_counts=camera_counts, cameras_found=cameras_found)


def analyze_observations(traj_dir: Path) -> TrajObsInfo:
    obs_path = traj_dir / "obs_dict.pkl"
    key_shapes: Dict[str, Tuple[int, ...]] = {}
    key_dtypes: Dict[str, str] = {}
    step_count: Optional[int] = None
    if obs_path.exists():
        try:
            with open(obs_path, "rb") as f:
                obs = pickle.load(f)
            if isinstance(obs, dict):
                for k, v in obs.items():
                    if hasattr(v, "shape"):
                        key_shapes[k] = tuple(int(s) for s in v.shape)
                        key_dtypes[k] = str(getattr(v, "dtype", ""))
                        if step_count is None and len(v.shape) > 0:
                            step_count = int(v.shape[0])
                    elif isinstance(v, list):
                        key_shapes[k] = (len(v),)
                        key_dtypes[k] = f"list[{type(v[0]).__name__ if v else 'Any'}]"
                        if step_count is None:
                            step_count = len(v)
                    else:
                        key_shapes[k] = ()
                        key_dtypes[k] = type(v).__name__
        except Exception:
            pass
    return TrajObsInfo(step_count=step_count, key_shapes=key_shapes, key_dtypes=key_dtypes)


def _stack_actions_from_policy_list(pol: List[Any]) -> Optional[np.ndarray]:
    """Extract actions as (T, D) array from policy_out list of dicts."""
    if not isinstance(pol, list) or not pol:
        return None
    rows: List[np.ndarray] = []
    for step in pol:
        if not isinstance(step, dict):
            continue
        act = step.get("actions")
        if act is None:
            continue
        if isinstance(act, np.ndarray):
            rows.append(act.astype(float))
        elif isinstance(act, (list, tuple)):
            rows.append(np.asarray(act, dtype=float))
    if not rows:
        return None
    # Pad ragged rows if needed
    max_d = max(r.shape[-1] for r in rows)
    pad_rows = []
    for r in rows:
        if r.shape[-1] == max_d:
            pad_rows.append(r)
        else:
            tmp = np.zeros((max_d,), dtype=float)
            tmp[: r.shape[-1]] = r
            pad_rows.append(tmp)
    return np.stack(pad_rows, axis=0)


def analyze_actions(traj_dir: Path) -> TrajActionInfo:
    pol_path = traj_dir / "policy_out.pkl"
    step_count: Optional[int] = None
    action_dim: Optional[int] = None
    policy_type: Optional[str] = None
    always_zero_dims: Optional[List[int]] = None
    min_per_dim: Optional[List[float]] = None
    max_per_dim: Optional[List[float]] = None
    if pol_path.exists():
        try:
            pol = safe_pickle_load(pol_path)
            if isinstance(pol, list):
                step_count = len(pol)
                arr = _stack_actions_from_policy_list(pol)
                if arr is not None and arr.size > 0:
                    action_dim = int(arr.shape[-1])
                    # Stats for always-zero detection
                    eps = 1e-8
                    minv = np.min(arr, axis=0).tolist()
                    maxv = np.max(arr, axis=0).tolist()
                    zero_dims = [i for i in range(arr.shape[1]) if np.all(np.isclose(arr[:, i], 0.0, atol=eps))]
                    min_per_dim = [float(x) for x in minv]
                    max_per_dim = [float(x) for x in maxv]
                    always_zero_dims = zero_dims
                if step_count:
                    first = pol[0]
                    if isinstance(first, dict):
                        policy_type = str(first.get("policy_type")) if "policy_type" in first else None
        except Exception:
            pass
    return TrajActionInfo(
        step_count=step_count,
        action_dim=action_dim,
        policy_type=policy_type,
        always_zero_dims=always_zero_dims,
        min_per_dim=min_per_dim,
        max_per_dim=max_per_dim,
    )


def plot_action_curves(traj_dir: Path, out_dir: Path, title: str) -> Optional[Path]:
    """Generate a single-axes plot with all action dims over time.

    Returns the saved image path or None if plotting not possible.
    """
    pol_path = traj_dir / "policy_out.pkl"
    if not pol_path.exists():
        return None
    try:
        pol = safe_pickle_load(pol_path)
        arr = _stack_actions_from_policy_list(pol)
        if arr is None or arr.size == 0:
            return None
        T, D = arr.shape
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{title}.png"
        plt.figure(figsize=(10, 4))
        for d in range(D):
            plt.plot(range(T), arr[:, d], label=f"a{d}", linewidth=1.2)
        plt.title(title)
        plt.xlabel("t")
        plt.ylabel("action value")
        plt.grid(True, alpha=0.3)
        if D <= 10:
            plt.legend(ncol=min(D, 7), fontsize=8)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path
    except Exception:
        return None


def analyze_single_traj(task: str, timestamp: str, traj_group: str, traj_name: str, traj_dir: Path) -> TrajAnalysis:
    images = analyze_images(traj_dir)
    observations = analyze_observations(traj_dir)
    actions = analyze_actions(traj_dir)

    warnings: List[str] = []
    # Compare counts where available
    img_counts = list(images.camera_counts.values())
    if img_counts and observations.step_count is not None:
        # Frames often equal to obs steps; sometimes off-by-one depending on logging
        min_imgs = min(img_counts)
        if min_imgs not in {observations.step_count, observations.step_count - 1, observations.step_count + 1}:
            warnings.append(
                f"Image/obs mismatch: min_images={min_imgs}, obs_steps={observations.step_count}"
            )
    if actions.step_count is not None and observations.step_count is not None:
        if actions.step_count not in {observations.step_count, observations.step_count - 1, observations.step_count + 1}:
            warnings.append(
                f"Action/obs mismatch: action_steps={actions.step_count}, obs_steps={observations.step_count}"
            )

    return TrajAnalysis(
        task=task,
        timestamp=timestamp,
        traj_group=traj_group,
        traj_name=traj_name,
        traj_path=str(traj_dir),
        images=images,
        observations=observations,
        actions=actions,
        warnings=warnings,
    )


def scan_dataset(dataset_root: Path, task: Optional[str] = None, limit: Optional[int] = None,
                 plot_actions: bool = False, actions_out_dir: Optional[Path] = None) -> Dict[str, Any]:
    dataset_root = Path(dataset_root)
    assert dataset_root.exists(), f"Dataset root not found: {dataset_root}"

    tasks = [task] if task else [d.name for d in dataset_root.iterdir() if d.is_dir()]
    tasks.sort()

    results: List[Dict[str, Any]] = []
    totals = {
        "tasks": len(tasks),
        "sessions": 0,
        "traj_groups": 0,
        "trajectories": 0,
    }

    analyzed = 0
    for t in tasks:
        t_dir = dataset_root / t
        if not t_dir.is_dir():
            continue
        sessions = [d.name for d in t_dir.iterdir() if d.is_dir()]
        sessions.sort()
        totals["sessions"] += len(sessions)

        for ts in sessions:
            ts_dir = t_dir / ts / "raw"
            if not ts_dir.exists():
                continue
            groups = [d.name for d in ts_dir.iterdir() if d.is_dir() and d.name.startswith("traj_group")]
            groups.sort()
            totals["traj_groups"] += len(groups)
            for g in groups:
                g_dir = ts_dir / g
                trajs = [d.name for d in g_dir.iterdir() if d.is_dir() and d.name.startswith("traj")]
                trajs.sort(key=lambda n: int(n.replace("traj", "")) if n.replace("traj", "").isdigit() else n)
                totals["trajectories"] += len(trajs)

                for tr in trajs:
                    traj_dir = g_dir / tr
                    analysis = analyze_single_traj(t, ts, g, tr, traj_dir)
                    # Optional plotting of action curves
                    if plot_actions and actions_out_dir is not None:
                        title = f"{t}_{ts}_{tr}"
                        img_path = plot_action_curves(traj_dir, actions_out_dir, title)
                        if img_path is not None:
                            analysis.actions.curve_image_path = str(img_path)
                    results.append(asdict(analysis))
                    analyzed += 1
                    if limit is not None and analyzed >= limit:
                        return {
                            "dataset_root": str(dataset_root),
                            "totals": totals,
                            "samples": results,
                        }

    return {
        "dataset_root": str(dataset_root),
        "totals": totals,
        "samples": results,
    }


def main():  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Analyze BridgeDataset v2-style dataset structure and signal dimensions")
    parser.add_argument("--dataset-root", default="/path/to/dataset/bdv2", help="Dataset root directory")
    parser.add_argument("--task", default=None, help="Optional task folder to restrict analysis (e.g., open_microwave)")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of trajectories analyzed (None = all)")
    parser.add_argument("--save-json", default=str(Path(__file__).parent / "bdv2_inspect" / "analysis_summary.json"), help="Where to save JSON summary")
    parser.add_argument("--plot-actions", action="store_true", help="Generate action curves for scanned trajectories")
    parser.add_argument("--actions-out-dir", default=str(Path(__file__).parent / "bdv2_inspect" / "actions"), help="Directory to save action curve images")
    parser.add_argument("--print-details", action="store_true", help="Print per-trajectory details to stdout")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    summary = scan_dataset(
        dataset_root,
        task=args.task,
        limit=None if args.limit == -1 else args.limit,
        plot_actions=args.plot_actions,
        actions_out_dir=Path(args.actions_out_dir),
    )

    # Ensure output dir exists
    out_path = Path(args.save_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    totals = summary["totals"]
    print("BDV2 Dataset Analysis Summary")
    print(f"  Root: {summary['dataset_root']}")
    print(f"  Tasks: {totals['tasks']}  Sessions: {totals['sessions']}  Groups: {totals['traj_groups']}  Trajs: {totals['trajectories']}")
    print(f"  Saved JSON: {out_path}")

    if args.print_details:
        for s in summary["samples"]:
            imgs = s["images"]["camera_counts"]
            obs = s["observations"]["key_shapes"]
            act_steps = s["actions"]["step_count"]
            act_dim = s["actions"]["action_dim"]
            warn = s["warnings"]
            print(f"- {s['task']}/{s['timestamp']}/{s['traj_group']}/{s['traj_name']}")
            print(f"    images: {imgs}")
            if obs:
                # Print select keys succinctly
                select_keys = ["state", "full_state", "qpos", "qvel", "eef_transform"]
                for k in select_keys:
                    if k in obs:
                        print(f"    obs[{k}]: {tuple(obs[k])}")
            az = s["actions"].get("always_zero_dims")
            minv = s["actions"].get("min_per_dim")
            maxv = s["actions"].get("max_per_dim")
            curve = s["actions"].get("curve_image_path")
            print(f"    actions: steps={act_steps}, dim={act_dim}")
            if minv is not None and maxv is not None:
                print(f"    action min: {[round(x,4) for x in minv]}")
                print(f"    action max: {[round(x,4) for x in maxv]}")
            if az:
                print(f"    always_zero_dims: {az}")
            if curve:
                print(f"    action_curve: {curve}")
            if warn:
                print(f"    warnings: {warn}")


if __name__ == "__main__":  # pragma: no cover
    main()

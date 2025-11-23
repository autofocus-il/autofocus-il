from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running both as a module (python -m bridge_torch.data.dataset_inspect)
# and as a standalone script (python bridge_torch/data/dataset_inspect.py)
from bridge_numpy import BridgeDataset  # type: ignore


def _coerce_paths(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            cand = os.path.join(p, "out.npy")
            if os.path.isfile(cand):
                out.append(cand)
        elif os.path.isfile(p) and p.endswith(".npy"):
            out.append(p)
    if not out:
        raise FileNotFoundError("No valid .npy paths found from --paths arguments")
    return out


def summarize_batch(batch: Dict[str, Any]) -> None:
    def _tinfo(name: str, t: torch.Tensor):
        print(f"- {name}: torch.Tensor, shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")
        try:
            with torch.no_grad():
                mn = float(t.min().item())
                mx = float(t.max().item())
            print(f"  range: [{mn:.6g}, {mx:.6g}]")
        except Exception:
            pass

    print("Batch summary:")
    obs = batch.get("observations", {})
    goals = batch.get("goals", {})
    acts = batch.get("actions", None)

    if isinstance(obs, dict):
        img = obs.get("image")
        if isinstance(img, torch.Tensor):
            _tinfo("observations.image", img)
        prop = obs.get("proprio")
        if isinstance(prop, torch.Tensor):
            _tinfo("observations.proprio", prop)
        sal = obs.get("saliency")
        if isinstance(sal, torch.Tensor):
            _tinfo("observations.saliency", sal)

    if isinstance(goals, dict):
        gimg = goals.get("image")
        if isinstance(gimg, torch.Tensor):
            _tinfo("goals.image", gimg)

    if isinstance(acts, torch.Tensor):
        _tinfo("actions", acts)
    elif isinstance(acts, np.ndarray):
        print(f"- actions: np.ndarray, shape={acts.shape}, dtype={acts.dtype}")
        try:
            print(f"  range: [{acts.min():.6g}, {acts.max():.6g}]")
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(
        description="Build BridgeDataset and fetch a single batch for inspection",
    )
    p.add_argument("--paths", nargs="+", default=["/path/to/test_dataset/bdv2_numpy/lift_carrot_100/train"],
                   help="One or more directories containing out.npy or direct out.npy paths")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", action="store_true", help="Enable train mode (affects augmentation)")
    p.add_argument("--obs_horizon", type=int, default=1)
    p.add_argument("--act_pred_horizon", type=int, default=1)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--augment_next_obs_goal_differently", action="store_true")
    p.add_argument("--saliency_alpha", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", type=int, default=1)

    args = p.parse_args()

    npy_paths = _coerce_paths(args.paths)

    ds = BridgeDataset(
        data_paths=npy_paths,
        seed=args.seed,
        train=bool(args.train),
        goal_relabeling_strategy="uniform",
        goal_relabeling_kwargs=None,
        relabel_actions=True,
        augment=bool(args.augment),
        augment_next_obs_goal_differently=bool(args.augment_next_obs_goal_differently),
        act_pred_horizon=int(args.act_pred_horizon),
        obs_horizon=int(args.obs_horizon),
        augment_kwargs=None,
        saliency_alpha=float(args.saliency_alpha),
    )

    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=False,
    )

    it = iter(dl)
    batch = next(it)
    summarize_batch(batch)


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
import albumentations as A
import cv2
import ast


def _to_pair(v, symm=False, around1=False):
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            if symm:
                a = -float(v[0]); b = float(v[0])
                return (a, b)
            if around1:
                a = 1.0 - float(v[0]); b = 1.0 + float(v[0])
                return (a, b)
            return (0.0, float(v[0]))
        return (float(v[0]), float(v[1]))
    return (float(v), float(v))


class BridgeDataset(Dataset):
    """Torch-style Dataset for BridgeData (bdv2 numpy out.npy files).

    - Loads all trajectories listed in `data_paths` (each an `out.npy`).
    - Flattens valid (traj, t) pairs into an index, respecting `obs_horizon` and `act_pred_horizon`.
    - Returns a dict of tensors per sample.
    """

    def __init__(
        self,
        data_paths: List[str],
        *,
        seed: int,
        train: bool,
        goal_relabeling_strategy: str = "uniform",
        goal_relabeling_kwargs: Optional[dict] = None,
        relabel_actions: bool = True,
        augment: bool = False,
        augment_next_obs_goal_differently: bool = False,
        act_pred_horizon: Optional[int] = None,
        obs_horizon: Optional[int] = None,
        augment_kwargs: Optional[dict] = None,
        saliency_alpha: Optional[float] = None,
        action_proprio_metadata: Optional[dict] = None,
        sample_weights: Optional[List[float]] = None,
        load_images: bool = True,
        **_: dict,
    ) -> None:
        super().__init__()
        self.rng = random.Random(int(seed))
        self._base_seed = int(seed)
        self.train = bool(train)
        self.goal_strategy = str(goal_relabeling_strategy or "uniform")
        self.goal_kwargs = dict(goal_relabeling_kwargs or {})
        self.relabel_actions = bool(relabel_actions)
        self.obs_h = int(obs_horizon) if obs_horizon is not None else 1
        self.act_h = int(act_pred_horizon) if act_pred_horizon is not None else 1
        self.saliency_alpha = float(saliency_alpha) if saliency_alpha is not None else 1.0
        self.augment = bool(augment)
        self.augment_next_obs_goal_differently = bool(augment_next_obs_goal_differently)
        self.apm = action_proprio_metadata or {}
        self.load_images = bool(load_images)
        
        self.augment_kwargs = augment_kwargs
        self._albu_cache: dict[tuple[int, int], A.ReplayCompose] = {}

        # Load all trajectories
        self.trajs: List[dict] = []
        for p in data_paths:
            with open(p, "rb") as f:
                trajs = np.load(f, allow_pickle=True)
            self.trajs.extend(list(trajs))

        # Build flat index of valid (traj, t)
        self._index: List[Tuple[int, int]] = []
        self._traj_lengths: List[int] = []
        for i, tr in enumerate(self.trajs):
            T = int(len(tr["actions"]))
            self._traj_lengths.append(T)
            # time aligned pairs exist for t in [0..T-1], using obs[t] and next_obs[t]
            past = self.obs_h - 1
            future = self.act_h - 1
            lo = max(0, past)
            hi = max(lo, T - 1 - future)  # inclusive
            for t in range(lo, hi + 1):
                self._index.append((i, t))

        if not self._index:
            raise ValueError("No valid samples found in provided data paths")

    def reseed(self, seed: int) -> None:
        """Reset RNG for multi-worker settings."""
        self.rng = random.Random(int(seed))

    # ----- Albumentations helpers -----
    def _build_albu(self, H: int, W: int) -> A.ReplayCompose:
        ak = self.augment_kwargs or {}
        order = ak.get(
            "augment_order",
            [
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        )
        p = ak.get("p", 0.5)
        ops: List[A.BasicTransform] = []
        for op in order:
            if op == "random_resized_crop" and ("random_resized_crop" in ak):
                rrc = ak["random_resized_crop"]
                scale = tuple(rrc.get("scale", [0.8, 1.0]))
                ratio = tuple(rrc.get("ratio", [0.9, 1.1]))
                ops.append(A.RandomResizedCrop(size=(H, W), scale=scale, ratio=ratio, interpolation=cv2.INTER_LINEAR))
            elif op == "random_brightness" and ("random_brightness" in ak):
                br = _to_pair(ak["random_brightness"], around1=True)
                ops.append(A.ColorJitter(brightness=br, contrast=0, saturation=0, hue=0))
            elif op == "random_contrast" and ("random_contrast" in ak):
                ct = _to_pair(ak["random_contrast"], around1=True)
                ops.append(A.ColorJitter(brightness=0, contrast=ct, saturation=0, hue=0))
            elif op == "random_saturation" and ("random_saturation" in ak):
                st = _to_pair(ak["random_saturation"], around1=True)
                ops.append(A.ColorJitter(brightness=0, contrast=0, saturation=st, hue=0))
            elif op == "random_hue" and ("random_hue" in ak):
                hu = _to_pair(ak["random_hue"], symm=True)
                ops.append(A.ColorJitter(brightness=0, contrast=0, saturation=0, hue=hu))
            else:
                continue
        return A.ReplayCompose(ops, p=p)

    # ----- Builders -----
    def _make_obs_image(self, traj: dict, t: int, batch_replay: Optional[dict] = None) -> Tuple[Optional[torch.Tensor], Optional[dict]]:
        # returns (C,H,W float[0,1], replay_params) or (None, None) if load_images=False
        if not self.load_images:
            return None, None
        if self.obs_h > 1:
            t0 = max(0, t - (self.obs_h - 1))
            frames = [traj["observations"][k]["images0"] for k in range(t0, t + 1)]  # THWC
            Tn, H, W, _ = len(frames), frames[0].shape[0], frames[0].shape[1], frames[0].shape[2]
            if self.train and self.augment:
                if batch_replay is not None:
                    replay = batch_replay
                    imgs = [A.ReplayCompose.replay(replay, image=f)["image"] for f in frames]
                else:
                    rc = self._albu_cache.get((H, W))
                    if rc is None:
                        rc = self._build_albu(H, W)
                        self._albu_cache[(H, W)] = rc
                    first = rc(image=frames[0])
                    replay = first["replay"]
                    imgs = [first["image"]]
                    for i in range(1, Tn):
                        imgs.append(A.ReplayCompose.replay(replay, image=frames[i])["image"])
            else:
                imgs = frames
                replay = None
            ts = [torch.as_tensor(im.transpose(2, 0, 1), dtype=torch.float32) / 255.0 for im in imgs]
            out = torch.cat(ts, dim=0)
            return out.contiguous(), replay
        else:
            img = traj["observations"][t]["images0"]  # HWC
            H, W = img.shape[:2]
            if self.train and self.augment:
                if batch_replay is not None:
                    replay = batch_replay
                    img = A.ReplayCompose.replay(replay, image=img)["image"]
                else:
                    rc = self._albu_cache.get((H, W))
                    if rc is None:
                        rc = self._build_albu(H, W)
                        self._albu_cache[(H, W)] = rc
                    out = rc(image=img)
                    img = out["image"]
                    replay = out["replay"]
            else:
                replay = None
            t_img = torch.as_tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            return t_img.contiguous(), replay

    def _make_goal_image(self, traj: dict, t: int, replay: Optional[dict]) -> Optional[torch.Tensor]:
        # pick goal image per strategy, returns None if load_images=False
        if not self.load_images:
            return None
        if self.goal_strategy == "uniform":
            T = len(traj["observations"]) - 1
            g_t = self.rng.randrange(t, T + 1)
            g = traj["next_observations"][g_t]["images0"]
        else:
            g = traj["next_observations"][t]["images0"]
        H, W = g.shape[:2]
        img = g
        if self.train and self.augment:
            if self.augment_next_obs_goal_differently or replay is None:
                rc = self._albu_cache.get((H, W))
                if rc is None:
                    rc = self._build_albu(H, W)
                    self._albu_cache[(H, W)] = rc
                img = rc(image=img)["image"]
            else:
                img = A.ReplayCompose.replay(replay, image=img)["image"]
        t_img = torch.as_tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        return t_img.contiguous()

    def _make_saliency(self, traj: dict, t: int, replay: Optional[dict]) -> Optional[torch.Tensor]:
        if not self.load_images:
            return None
        obs = traj["observations"][t]
        if "saliency" not in obs or obs["saliency"] is None:
            return None
        if self.obs_h > 1:
            t0 = max(0, t - (self.obs_h - 1))
            alpha = float(self.saliency_alpha)
            agg = None
            for k in range(t, t0 - 1, -1):
                s = traj["observations"][k].get("saliency", None)
                if s is None:
                    continue
                d = t - k
                w = (alpha ** d) if d > 0 else 1.0
                cur = (w * s).astype(np.float32)
                agg = cur if agg is None else (agg + cur)
            if agg is None:
                agg = obs["saliency"].astype(np.float32)
            sal = agg  # HWC with C=1
        else:
            sal = obs["saliency"].astype(np.float32)
        # If augmented, apply the same spatial crop/resize to saliency via replay.
        # Treat saliency as a mask to avoid color jitter.
        if self.train and self.augment and replay is not None:
            img_ref = obs["images0"]  # HWC reference image for consistent replay
            out = A.ReplayCompose.replay(replay, image=img_ref, mask=sal)
            sal = out["mask"]
            if sal.ndim == 2:
                sal = sal[..., None]
        t_sal = torch.as_tensor(sal.transpose(2, 0, 1), dtype=torch.float32)
        # normalize to [0,1]
        mn = t_sal.amin(dim=(-2, -1), keepdim=True)
        mx = t_sal.amax(dim=(-2, -1), keepdim=True)
        t_sal = (t_sal - mn) / (mx - mn + 1e-8)
        return t_sal.contiguous()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        ti, t = self._index[idx]
        traj = self.trajs[ti]

        obs_img, replay = self._make_obs_image(traj, t)
        goal_img = self._make_goal_image(traj, t, replay)
        sal = self._make_saliency(traj, t, replay)

        # proprio features
        if self.obs_h > 1:
            t0 = max(0, t - (self.obs_h - 1))
            prop_seq = None
            if traj["observations"][t].get("state", None) is not None:
                prop_seq = np.stack([traj["observations"][k].get("state", None) for k in range(t0, t + 1)], axis=0)
        else:
            prop_seq = traj["observations"][t].get("state", None)

        # actions (sequence if act_h>1)
        if self.act_h > 1:
            a_T = len(traj["actions"])  # same as len(observations)
            t_end = min(a_T - 1, t + (self.act_h - 1))
            acts = np.stack([traj["actions"][k] for k in range(t, t_end + 1)], axis=0).astype(np.float32)
        else:
            acts = np.asarray(traj["actions"][t], dtype=np.float32)

        obs_dict: Dict[str, torch.Tensor] = {}
        if obs_img is not None:
            obs_dict["image"] = obs_img.contiguous()
        if prop_seq is not None:
            p = np.asarray(prop_seq, dtype=np.float32)
            if p.ndim > 1:
                p = p.reshape(-1)
            obs_dict["proprio"] = torch.from_numpy(p)
        if sal is not None:
            obs_dict["saliency"] = sal.contiguous()

        # tile goal channels if obs has T>1 (C != 3)
        g = goal_img
        if g is not None and obs_img is not None:
            if g.shape[0] != obs_img.shape[0]:
                if g.shape[0] == 3 and obs_img.shape[0] % 3 == 0:
                    k = obs_img.shape[0] // 3
                    g = g.repeat(k, 1, 1)
                else:
                    # simple pad/trim fallback
                    if g.shape[0] < obs_img.shape[0]:
                        pad = obs_img.shape[0] - g.shape[0]
                        g = torch.cat([g, g[:pad]], dim=0)
                    else:
                        g = g[:obs_img.shape[0]]

        out = {
            "observations": obs_dict,
            "actions": torch.from_numpy(acts),
        }
        if g is not None:
            out["goals"] = {"image": g.contiguous()}
        return out

    def iterator(self, batch_size: int):
        """Yield full batches by sampling random (traj,t) pairs.

        - Shares a single Albumentations replay across the batch for consistency.
        - Applies pin_memory on the batch tensors for faster H2D copies.
        """
        n = len(self._index)
        assert n > 0, "Empty dataset"
        while True:
            # Sample indices with replacement
            sel = [self.rng.randrange(n) for _ in range(batch_size)]
            pairs = [self._index[i] for i in sel]

            # Build a shared replay for the batch (based on first sample's shape)
            batch_replay = None
            if self.train and self.augment:
                ti0, t0 = pairs[0]
                obs0 = self.trajs[ti0]["observations"][t0]["images0"]
                H0, W0 = obs0.shape[:2]
                rc = self._albu_cache.get((H0, W0))
                if rc is None:
                    rc = self._build_albu(H0, W0)
                    self._albu_cache[(H0, W0)] = rc
                # trigger a sample to capture replay params
                batch_replay = rc(image=obs0)["replay"]

            obs_imgs: List[torch.Tensor] = []
            goal_imgs: List[torch.Tensor] = []
            sal_list: List[torch.Tensor] = []
            prop_list: List[Optional[np.ndarray]] = []
            act_list: List[np.ndarray] = []

            for (ti, t) in pairs:
                traj = self.trajs[ti]
                oi, rep = self._make_obs_image(traj, t, batch_replay=batch_replay)
                gi = self._make_goal_image(traj, t, rep if not self.augment_next_obs_goal_differently else None)
                si = self._make_saliency(traj, t, rep)

                # proprio
                if self.obs_h > 1:
                    t0 = max(0, t - (self.obs_h - 1))
                    prop_seq = None
                    if traj["observations"][t].get("state", None) is not None:
                        prop_seq = np.stack([traj["observations"][k].get("state", None) for k in range(t0, t + 1)], axis=0)
                else:
                    prop_seq = traj["observations"][t].get("state", None)

                # actions
                if self.act_h > 1:
                    a_T = len(traj["actions"])  # same as len(observations)
                    t_end = min(a_T - 1, t + (self.act_h - 1))
                    acts = np.stack([traj["actions"][k] for k in range(t, t_end + 1)], axis=0).astype(np.float32)
                else:
                    acts = np.asarray(traj["actions"][t], dtype=np.float32)

                obs_imgs.append(oi)
                goal_imgs.append(gi)
                if si is not None:
                    sal_list.append(si)
                prop_list.append(prop_seq)
                act_list.append(acts)

            # Stack batch tensors
            obs_img_t = torch.stack(obs_imgs, dim=0) if obs_imgs and obs_imgs[0] is not None else None
            goal_img_t = torch.stack(goal_imgs, dim=0) if goal_imgs and goal_imgs[0] is not None else None
            actions_t = torch.from_numpy(np.stack(act_list, axis=0))

            obs_dict: Dict[str, torch.Tensor] = {}
            if obs_img_t is not None:
                obs_dict["image"] = obs_img_t
            # proprio stacking
            if any(p is not None for p in prop_list):
                # fill missing with zeros like shape of first non-None
                shapes = [np.asarray(p).reshape(-1).shape[0] for p in prop_list if p is not None]
                if shapes:
                    P = shapes[0]
                    flat_props = []
                    for p in prop_list:
                        if p is None:
                            flat_props.append(np.zeros((P,), dtype=np.float32))
                        else:
                            q = np.asarray(p, dtype=np.float32)
                            if q.ndim > 1:
                                q = q.reshape(-1)
                            flat_props.append(q)
                    obs_dict["proprio"] = torch.from_numpy(np.stack(flat_props, axis=0))

            if sal_list and len(sal_list) == len(pairs):
                obs_dict["saliency"] = torch.stack(sal_list, dim=0)

            batch = {
                "observations": obs_dict,
                "actions": actions_t,
            }
            if goal_img_t is not None:
                batch["goals"] = {"image": goal_img_t}

            # Pin memory for faster H2D copies
            try:
                if "image" in batch["observations"]:
                    batch["observations"]["image"] = batch["observations"]["image"].pin_memory()
                if "proprio" in batch["observations"]:
                    batch["observations"]["proprio"] = batch["observations"]["proprio"].pin_memory()
                if "saliency" in batch["observations"]:
                    batch["observations"]["saliency"] = batch["observations"]["saliency"].pin_memory()
                if "goals" in batch and "image" in batch["goals"]:
                    batch["goals"]["image"] = batch["goals"]["image"].pin_memory()
                batch["actions"] = batch["actions"].pin_memory()
            except Exception:
                pass

            yield batch


def build_bridge_dataset(
    task_globs: List[Union[str, List[str]]],
    *,
    data_root: str,
    split: str,
    seed: int,
    bridgedata_cfg,
    dataset_kwargs: Dict,
    load_images: bool = True,
) -> BridgeDataset:
    # Resolve directories that contain out.npy
    inc = getattr(bridgedata_cfg, "include", [])
    try:
        inc_list = list(inc) if inc is not None else []
    except Exception:
        inc_list = inc if isinstance(inc, (list, tuple)) else []

    groups: List[List[str]] = []
    if inc_list:
        first = inc_list[0]
        if isinstance(first, (str, bytes)):
            groups = [[str(x) for x in inc_list]]
        else:
            groups = [[str(x) for x in list(g)] for g in inc_list]
    task_files: List[str] = []
    for sub in groups:
        for name in sub:
            task_files.append(os.path.join(data_root, str(name), split, "out.npy"))

    if not task_files:
        raise ValueError(
            f"No out.npy found: include={inc_list}, data_root={data_root}, split={split}. "
            "Check bridgedata preset and data_path."
        )
    return BridgeDataset(task_files, seed=seed, train=(split == "train"), action_proprio_metadata=bridgedata_cfg.action_proprio_metadata, load_images=load_images, **dataset_kwargs)


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = 2,
    drop_last: bool = False,
    sampler=None,
    batch_mode: bool = True,
) -> DataLoader:
    if not batch_mode:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if (num_workers > 0 and prefetch_factor is not None) else None,
            drop_last=drop_last,
        )

    class _BatchIterable(IterableDataset):
        def __init__(self, ds: BridgeDataset, bs: int):
            super().__init__()
            self.ds = ds
            self.bs = bs

        def __iter__(self):
            wi = get_worker_info()
            if wi is not None:
                self.ds.reseed(self.ds._base_seed + 997 * int(wi.id))
            it = self.ds.iterator(self.bs)
            for b in it:
                yield b

    iterable = _BatchIterable(dataset, batch_size)
    def _id_collate(x):
        return x[0] if isinstance(x, list) and len(x) == 1 else x

    return DataLoader(
        iterable,
        batch_size=None,
        shuffle=False,
        collate_fn=_id_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if (num_workers > 0 and prefetch_factor is not None) else None,
    )

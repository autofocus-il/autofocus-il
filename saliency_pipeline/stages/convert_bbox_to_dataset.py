#!/usr/bin/env python3
"""
Convert BBoxes To Dataset (Only for CARLA)

Reads per-frame bounding boxes from saliency_exp_results JSON files and saves
them into the Bench2Drive dataset structure as .pt tensors:

- From grounding_detections.json -> non_filter.pt
- From vlm_filtered_boxes.json   -> filter_dynamic.pt

Reference: gaze_pseudo_generator.py (structure, YAML-driven flow).
"""

from __future__ import annotations

import json
import pickle
import yaml
import torch
import numpy as np
import cv2
import imageio
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import argparse
import random

# Reuse pipeline utilities for route/seed enumeration
# Local module import (same dir)
from pathlib import Path

from saliency_pipeline.utils.pipeline_utils import get_routes_seeds, load_pipeline_config
from saliency_pipeline.utils.config_manager import load_merged_config
_GIF_ALLOWED: set[str] = set()
_GIF_ROOT: Optional[Path] = None


def load_config(config_path: Optional[Path] = None, domain: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration using two-domain layout (merging common + domain).

    Falls back to legacy layout if provided.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "bbox_to_dataset_config.yaml"
    return load_merged_config(Path(config_path), domain=domain)


def enumerate_pairs(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return list of (route, seed) strings to process, based on run_mode.

    Uses routes_seeds.pairs from refactor/configs/pipeline_config.yaml when mode == 'all'.
    """
    run_cfg = cfg.get("run_mode", {})
    mode = (run_cfg.get("mode") or "all").lower()

    def to_route(rid: int) -> str:
        return f"route_{rid}"

    def to_seed(sid: int) -> str:
        return f"seed_{sid}"

    if mode == "single_seed":
        ss = run_cfg.get("single_seed", {})
        route = to_route(int(ss["route_id"]))
        seed = to_seed(int(ss["seed_id"]))
        return [(route, seed)]
    elif mode == "single_route":
        sr = run_cfg.get("single_route", {})
        rid = int(sr["route_id"])  # required
        # Try to pull seeds for this route from shared pipeline config
        try:
            _ = load_pipeline_config()  # ensure loaded
            pairs = [p for p in get_routes_seeds() if p[0] == rid]
            seeds = sorted({s for (_, s) in pairs})
        except Exception:
            seeds = []
        # Fallback to 200..219 if not found
        if not seeds:
            seeds = list(range(200, 220))
        return [(to_route(rid), to_seed(s)) for s in seeds]
    elif mode == "all":
        # Enumerate all pairs from shared pipeline configuration
        _ = load_pipeline_config()
        pairs = get_routes_seeds()
        return [(to_route(r), to_seed(s)) for (r, s) in pairs]
    else:
        raise ValueError(f"Invalid run_mode: {mode}")


def enumerate_bdv2_targets(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Enumerate BDV2 targets. Returns list of dicts with keys:
    - route: route label used in saliency results (bdv2/...)
    - seed: seed label (images{camera})
    - traj_dir: path to BDV2 trajectory directory containing obs_dict.pkl
    """
    rm = cfg.get("run_mode", {}) or cfg.get("run", {})
    mode = rm.get("mode", "bdv2_single")
    # dataset_root may live under cfg['bdv2'] or cfg['dataset']['bdv2']
    bd_root_cfg = cfg.get("bdv2", {})
    if not bd_root_cfg:
        bd_root_cfg = cfg.get("dataset", {}).get("bdv2", {})
    dataset_root = Path(bd_root_cfg.get("dataset_root", "/path/to/dataset/bdv2"))
    out: List[Dict[str, Any]] = []
    if mode == "bdv2_single":
        single = rm.get("bdv2_single", {})
        task = single["task"]
        timestamp = single["timestamp"]
        traj_group = single["traj_group"]
        traj_name = single["traj_name"]
        camera = int(single.get("camera", 0))
        traj_dir = dataset_root / task / timestamp / "raw" / traj_group / traj_name
        route = f"bdv2/{task}/{timestamp}/{traj_group}/{traj_name}"
        seed = f"images{camera}"
        out.append({"route": route, "seed": seed, "traj_dir": traj_dir, "task": task})
    elif mode == "bdv2_task":
        task = rm.get("bdv2_task", {}).get("task")
        camera = int(rm.get("bdv2_task", {}).get("camera", 0))
        if not task:
            raise ValueError("bdv2_task.task is required")
        for ts_dir in sorted((dataset_root / task).iterdir()):
            raw = ts_dir / "raw"
            if not raw.exists():
                continue
            for g_dir in sorted(d for d in raw.iterdir() if d.is_dir() and d.name.startswith("traj_group")):
                for tr_dir in sorted(d for d in g_dir.iterdir() if d.is_dir() and d.name.startswith("traj")):
                    route = f"bdv2/{task}/{ts_dir.name}/{g_dir.name}/{tr_dir.name}"
                    seed = f"images{camera}"
                    out.append({"route": route, "seed": seed, "traj_dir": tr_dir, "task": task})
    elif mode == "bdv2_all":
        camera = int(rm.get("bdv2_all", {}).get("camera", 0))
        # Iterate all tasks under dataset_root
        for task_dir in sorted(d for d in dataset_root.iterdir() if d.is_dir()):
            task = task_dir.name
            for ts_dir in sorted((dataset_root / task).iterdir()):
                raw = ts_dir / "raw"
                if not raw.exists():
                    continue
                for g_dir in sorted(d for d in raw.iterdir() if d.is_dir() and d.name.startswith("traj_group")):
                    for tr_dir in sorted(d for d in g_dir.iterdir() if d.is_dir() and d.name.startswith("traj")):
                        route = f"bdv2/{task}/{ts_dir.name}/{g_dir.name}/{tr_dir.name}"
                        seed = f"images{camera}"
                        # task from directory name
                        out.append({"route": route, "seed": seed, "traj_dir": tr_dir, "task": task})
    else:
        raise ValueError(f"Invalid bdv2 run_mode: {mode}")
    return out


def load_grounding_boxes(json_path: Path) -> Tuple[List[List[List[float]]], int]:
    """Load per-frame bounding boxes from grounding_detections.json.

    Returns (boxes_per_frame, total_frames)
    boxes_per_frame is a list where each element is a list of [x1,y1,x2,y2] floats.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frame_entries = data.get("frame_detections", [])
    # Build frame -> list of bboxes map
    fmap: Dict[int, List[List[float]]] = {}
    for entry in frame_entries:
        idx = int(entry.get("frame_idx", 0))
        dets = entry.get("detections", []) or []
        bboxes = []
        for det in dets:
            box = det.get("bbox")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                bboxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
        fmap[idx] = bboxes

    total_frames = int(data.get("frames_processed") or (max(fmap.keys()) + 1 if fmap else 0))
    boxes_per_frame: List[List[List[float]]] = [fmap.get(i, []) for i in range(total_frames)]
    return boxes_per_frame, total_frames


def load_filtered_boxes(json_path: Path) -> Tuple[List[List[List[float]]], int]:
    """Load per-frame bounding boxes from vlm_filtered_boxes.json (key 'results').

    Returns (boxes_per_frame, total_frames)
    boxes_per_frame is a list where each element is a list of [x1,y1,x2,y2] floats.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    fmap: Dict[int, List[List[float]]] = {}
    for res in results:
        idx = int(res.get("frame_idx", 0))
        filtered = res.get("filtered", []) or []
        bboxes = []
        for det in filtered:
            box = det.get("bbox")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                bboxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
        fmap[idx] = bboxes

    total_frames = int(data.get("frames_processed") or (max(fmap.keys()) + 1 if fmap else 0))
    boxes_per_frame: List[List[List[float]]] = [fmap.get(i, []) for i in range(total_frames)]
    return boxes_per_frame, total_frames


def save_pt(seq: List[List[List[float]]], out_path: Path, overwrite: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        print(f"  ⚠️  Exists, skip: {out_path}")
        return
    torch.save(seq, out_path)
    print(f"  💾 Saved: {out_path}  (frames={len(seq)})")


def summarize(seq: List[List[List[float]]]) -> str:
    frames = len(seq)
    total_boxes = sum(len(f) for f in seq)
    avg = (total_boxes / frames) if frames else 0.0
    return f"frames={frames}, total_boxes={total_boxes}, avg/frame={avg:.2f}"


def process_one(route: str, seed: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = cfg["data"]
    fn = data_cfg["filenames"]
    overwrite = cfg.get("processing", {}).get("overwrite", True)

    saliency_dir = Path(data_cfg["saliency_results_dir"]) / route / seed
    dataset_dir = Path(data_cfg["dataset_dir"]) / route / seed

    grounding_json = saliency_dir / fn["grounding_detections"]
    vlm_filtered_json = saliency_dir / fn["vlm_filtered_boxes"]

    result: Dict[str, Any] = {
        "route": route,
        "seed": seed,
        "success": True,
        "written": {},
        "errors": [],
    }

    print(f"\n🎯 {route}/{seed}")

    # 1) non_filter.pt from grounding_detections.json
    try:
        if not grounding_json.exists():
            raise FileNotFoundError(f"Missing {grounding_json}")
        non_filter_seq, _ = load_grounding_boxes(grounding_json)
        out_non_filter = dataset_dir / fn["non_filter_pt"]
        save_pt(non_filter_seq, out_non_filter, overwrite)
        result["written"]["non_filter_pt"] = str(out_non_filter)
        print(f"  📊 non_filter: {summarize(non_filter_seq)}")
    except Exception as e:
        msg = f"non_filter failed: {e}"
        print(f"  ❌ {msg}")
        result["success"] = False
        result["errors"].append(msg)

    # 2) filter_dynamic.pt from vlm_filtered_boxes.json (key 'results')
    try:
        if not vlm_filtered_json.exists():
            raise FileNotFoundError(f"Missing {vlm_filtered_json}")
        filt_seq, _ = load_filtered_boxes(vlm_filtered_json)
        out_filter = dataset_dir / fn["filter_dynamic_pt"]
        save_pt(filt_seq, out_filter, overwrite)
        result["written"]["filter_dynamic_pt"] = str(out_filter)
        print(f"  📊 filter_dynamic: {summarize(filt_seq)}")
    except Exception as e:
        msg = f"filter_dynamic failed: {e}"
        print(f"  ❌ {msg}")
        result["success"] = False
        result["errors"].append(msg)

    return result


def centers_from_boxes(seq: List[List[List[float]]]) -> List[List[List[float]]]:
    centers: List[List[List[float]]] = []
    for frame_boxes in seq:
        pts: List[List[float]] = []
        for x1, y1, x2, y2 in frame_boxes:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            pts.append([float(cx), float(cy)])
        centers.append(pts)
    return centers


def process_one_bdv2(target: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """For BDV2, write bbox sequences as PKL next to obs_dict.pkl.

    - grounding_detections.json  -> no_filter.pkl (filename configurable)
    - vlm_filtered_boxes.json    -> filter.pkl    (filename configurable)
    - Also keeps the previous center-coordinates dump (filtered_coordinates.pkl) for backward compatibility.
    """
    data_cfg = cfg["data"]
    fn = data_cfg["filenames"]
    overwrite = cfg.get("processing", {}).get("overwrite", True)
    saliency_dir = Path(data_cfg["saliency_results_dir"]) / target["route"] / target["seed"]
    vlm_filtered_json = saliency_dir / fn["vlm_filtered_boxes"]
    grounding_json = saliency_dir / fn["grounding_detections"]
    traj_dir: Path = Path(target["traj_dir"])  # contains obs_dict.pkl
    # Control whether to write bbox PKLs (default False)
    write_bbox_pkls: bool = bool(data_cfg.get("write_bbox_pkls", False))
    # Configurable PKL output filenames (used only if write_bbox_pkls=True)
    pkl_fn_cfg: Dict[str, str] = data_cfg.get("pkl_filenames", {}) if write_bbox_pkls else {}
    no_filter_name = pkl_fn_cfg.get("no_filter", "no_filter.pkl") if write_bbox_pkls else None
    filter_name = pkl_fn_cfg.get("filter", "filter.pkl") if write_bbox_pkls else None

    out_no_filter_pkl = (traj_dir / no_filter_name) if no_filter_name else None
    out_filter_pkl = (traj_dir / filter_name) if filter_name else None

    result = {"route": target["route"], "seed": target["seed"], "traj_dir": str(traj_dir), "success": True, "written": {}, "errors": []}

    # 1) Optionally write no_filter.pkl from grounding_detections.json
    if write_bbox_pkls and out_no_filter_pkl is not None:
        try:
            if not grounding_json.exists():
                raise FileNotFoundError(f"Missing {grounding_json}")
            non_filter_seq, _ = load_grounding_boxes(grounding_json)
            if out_no_filter_pkl.exists() and not overwrite:
                print(f"  ⚠️  Exists, skip: {out_no_filter_pkl}")
            else:
                with open(out_no_filter_pkl, "wb") as f:
                    pickle.dump(non_filter_seq, f)
                print(f"  💾 Saved (BDV2 no_filter): {out_no_filter_pkl}  (frames={len(non_filter_seq)})")
            result["written"]["no_filter_pkl"] = str(out_no_filter_pkl)
        except Exception as e:
            msg = f"bdv2 no_filter write failed: {e}"
            print(f"  ❌ {msg}")
            result["success"] = False
            result["errors"].append(msg)

    # 2) Optionally write filter.pkl from vlm_filtered_boxes.json
    if write_bbox_pkls and out_filter_pkl is not None:
        try:
            if not vlm_filtered_json.exists():
                print(f"  ℹ️  No VLM-filtered JSON found, skipping filter.pkl ({vlm_filtered_json})")
            else:
                filt_seq, _ = load_filtered_boxes(vlm_filtered_json)
                if out_filter_pkl.exists() and not overwrite:
                    print(f"  ⚠️  Exists, skip: {out_filter_pkl}")
                else:
                    with open(out_filter_pkl, "wb") as f:
                        pickle.dump(filt_seq, f)
                    print(f"  💾 Saved (BDV2 filter): {out_filter_pkl}  (frames={len(filt_seq)})")
                result["written"]["filter_pkl"] = str(out_filter_pkl)
        except Exception as e:
            msg = f"bdv2 filter write failed: {e}"
            print(f"  ❌ {msg}")
            result["success"] = False
            result["errors"].append(msg)

    # 3) Generate saliency_map.pkl using multi-point Gaussian per-frame (H=480, W=640 by default)
    try:
        # Re-load grounding boxes if not already available
        if not grounding_json.exists():
            raise FileNotFoundError(f"Missing {grounding_json}")
        box_seq, total_frames = load_grounding_boxes(grounding_json)

        # Derive centers per frame
        center_seq = centers_from_boxes(box_seq)

        # Resolve saliency params (prefer bbox_to_dataset.saliency, fallback to bdv2.saliency)
        b2d_sal: Dict[str, Any] = cfg.get("bbox_to_dataset", {}).get("saliency", {}) or {}
        legacy_sal: Dict[str, Any] = cfg.get("bdv2", {}).get("saliency", {}) or {}
        sal_cfg: Dict[str, Any] = b2d_sal if b2d_sal else legacy_sal
        H = int(sal_cfg.get("image_height", 480))
        W = int(sal_cfg.get("image_width", 640))
        sigma = float(sal_cfg.get("sigma", 30.0))
        out_name = sal_cfg.get("output_name", "saliency_map.pkl")

        # Build 1D Gaussian kernel
        ksize = int(4 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        x = torch.arange(ksize, dtype=torch.float32) - ksize // 2
        kernel_1d = torch.exp(-(x ** 2) / (2 * (sigma ** 2)))
        kernel_1d = kernel_1d / (kernel_1d.sum() + 1e-8)
        padding = ksize // 2
        k_row = kernel_1d.view(1, 1, -1, 1)
        k_col = k_row.permute(0, 1, 3, 2)

        # Allocate output (T,1,H,W)
        T = len(center_seq)
        saliency_np = np.zeros((T, 1, H, W), dtype=np.float32)

        # Generate per-frame saliency
        for t, pts in enumerate(center_seq):
            if not pts:
                continue
            delta = torch.zeros(1, 1, H, W, dtype=torch.float32)
            # Heuristic: detect normalized vs pixel coords
            is_normalized = True
            for cx, cy in pts:
                if cx > 2.0 or cy > 2.0:
                    is_normalized = False
                    break
            for cx, cy in pts:
                if is_normalized:
                    xi = int(min(max(round(cx * (W - 1)), 0), W - 1))
                    yi = int(min(max(round(cy * (H - 1)), 0), H - 1))
                else:
                    xi = int(min(max(round(cx), 0), W - 1))
                    yi = int(min(max(round(cy), 0), H - 1))
                delta[0, 0, yi, xi] += 1.0

            blurred = torch.nn.functional.conv2d(delta, k_row, padding=(0, padding))
            blurred = torch.nn.functional.conv2d(blurred, k_col, padding=(padding, 0))
            min_v = blurred.amin(dim=(2, 3), keepdim=True)
            max_v = blurred.amax(dim=(2, 3), keepdim=True)
            norm = (blurred - min_v) / (max_v - min_v + 1e-8)
            saliency_np[t, 0] = norm[0, 0].numpy()

        out_saliency_pkl = traj_dir / out_name
        if out_saliency_pkl.exists() and not overwrite:
            print(f"  ⚠️  Exists, skip: {out_saliency_pkl}")
        else:
            with open(out_saliency_pkl, "wb") as f:
                pickle.dump(saliency_np, f)
            print(f"  💾 Saved (BDV2 saliency_map): {out_saliency_pkl}  (shape={saliency_np.shape}, dtype=float32)")
        result["written"]["saliency_map_pkl"] = str(out_saliency_pkl)

        # Build 50% overlay GIF for visualization (one random traj per task, unified directory)
        try:
            route = str(target.get("route", ""))
            if route in _GIF_ALLOWED and _GIF_ROOT is not None:
                # Locate image directory (seed like images0)
                img_dir = traj_dir / target["seed"]
                if not img_dir.exists():
                    # fallback: find first images* directory
                    cands = sorted([d for d in traj_dir.iterdir() if d.is_dir() and d.name.startswith("images")])
                    if cands:
                        img_dir = cands[0]
                img_paths = sorted(img_dir.glob("im_*.jpg"), key=lambda p: int(p.stem.split("_")[-1]))
                if not img_paths:
                    print(f"  ℹ️  No images found under {img_dir}, skip GIF overlay")
                else:
                    # Visualization config
                    viz_cfg: Dict[str, Any] = cfg.get("bbox_to_dataset", {}).get("visualization", {}) or {}
                    fps = int(viz_cfg.get("fps", 20))
                    # Build unified gif filename from route components
                    # route: bdv2/<task>/<timestamp>/<traj_group>/<traj_name>
                    parts = route.split("/")
                    task = parts[1] if len(parts) > 1 else "task"
                    timestamp = parts[2] if len(parts) > 2 else "ts"
                    traj_group = parts[3] if len(parts) > 3 else "g"
                    traj_name = parts[4] if len(parts) > 4 else "traj"
                    gif_name = f"{task}_{timestamp}_{traj_group}_{traj_name}_{target['seed']}.gif"
                    out_gif = _GIF_ROOT / gif_name
                    _GIF_ROOT.mkdir(parents=True, exist_ok=True)
                    frames = []
                    T_img = len(img_paths)
                    T = min(T_img, saliency_np.shape[0])
                    for t in range(T):
                        img_bgr = cv2.imread(str(img_paths[t]))
                        if img_bgr is None:
                            continue
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        H_img, W_img = img_rgb.shape[:2]
                        heat = saliency_np[t, 0]
                        # resize saliency to image size
                        heat_resized = cv2.resize((heat * 255.0).astype(np.uint8), (W_img, H_img), interpolation=cv2.INTER_LINEAR)
                        heat_color = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)
                        heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
                        overlay = (0.5 * img_rgb + 0.5 * heat_rgb).clip(0, 255).astype(np.uint8)
                        frames.append(overlay)
                    if frames:
                        imageio.mimsave(str(out_gif), frames, format="GIF", duration=max(1e-3, 1.0 / max(1, fps)))
                        print(f"  🎞️  Saved overlay GIF: {out_gif}  (frames={len(frames)}, fps={fps})")
                        result["written"]["saliency_gif"] = str(out_gif)
        except Exception as e:
            print(f"  ⚠️  Failed to write overlay GIF: {e}")
    except Exception as e:
        msg = f"bdv2 saliency_map write failed: {e}"
        print(f"  ❌ {msg}")
        result["success"] = False
        result["errors"].append(msg)


    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-frame bboxes from saliency_exp_results JSONs and save to dataset as .pt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                            # Use default YAML config
  %(prog)s --config refactor/configs/bbox_to_dataset_config.yaml
        """,
    )
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "bbox_to_dataset_config.yaml"),
                        help="Path to YAML config for this script (two-domain layout supported)")
    parser.add_argument("--domain", choices=["bench2drive", "bdv2"], default=None,
                        help="Override active domain in config (optional)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config), domain=args.domain)

    print("🚀 Convert BBoxes To Dataset")
    print("=" * 60)
    print(f"📂 saliency_results_dir: {cfg['data']['saliency_results_dir']}")
    ds_dir = cfg.get('data', {}).get('dataset_dir')
    if ds_dir:
        print(f"📁 dataset_dir: {ds_dir}")

    if cfg.get("dataset", {}).get("type", "bench2drive").lower() == "bdv2":
        targets = enumerate_bdv2_targets(cfg)
        # Decide random gif per task
        global _GIF_ALLOWED, _GIF_ROOT
        _GIF_ALLOWED = set()
        _GIF_ROOT = None
        try:
            viz_cfg = cfg.get("common", {}).get("data", {}).get("visualization", {}) or {}
            root = viz_cfg.get("gif_root", "/path/to/bdv2_saliency_filter_results/saliency_gifs")
            _GIF_ROOT = Path(root)
        except Exception:
            _GIF_ROOT = Path("/path/to/bdv2_saliency_filter_results/saliency_gifs")
        # group by task
        by_task: Dict[str, List[Dict[str, Any]]] = {}
        for t in targets:
            task = t.get("task") or (t.get("route", "").split("/")[1] if "/" in t.get("route", "") else "task")
            by_task.setdefault(task, []).append(t)
        for task, lst in by_task.items():
            choice = random.choice(lst)
            _GIF_ALLOWED.add(str(choice.get("route", "")))
        print(f"🔎 BDV2 targets: {len(targets)} trajs")
        ok = 0
        for t in targets:
            res = process_one_bdv2(t, cfg)
            if res.get("success"):
                ok += 1
        print("\n🎉 BDV2 conversion completed")
        print(f"   Success: {ok}/{len(targets)}")
        if ok < len(targets):
            print("   ⚠️  Some targets failed — see errors above.")
    else:
        pairs = enumerate_pairs(cfg)
        print(f"🔎 Targets: {len(pairs)} route/seed pairs")
        success = 0
        for route, seed in pairs:
            res = process_one(route, seed, cfg)
            if res.get("success"):
                success += 1
        print("\n🎉 Conversion completed")
        print(f"   Success: {success}/{len(pairs)}")
        if success < len(pairs):
            print("   ⚠️  Some pairs failed — see errors above.")


if __name__ == "__main__":
    main()

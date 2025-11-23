#!/usr/bin/env python3
"""
Explore Dataset - Bench2Drive Structure and File Inspector

This tool explores the Bench2Drive directory structure and inspects
per-seed .pt files including images and additional annotation tensors
like bbox centers (e.g., filter_dynamic.pt, non_filter.pt, gaze*.pt).

Usage examples:
  python -m carla_saliency.data_utils.explore_dataset \
      --data-dir /path/to/dataset/bench2drive220 \
      --route route_2416 --seed seed_200

  python -m carla_saliency.data_utils.explore_dataset --list-only
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def robust_load_pt(path: Path) -> Any:
    """Load a .pt that might contain tensor, ndarray, list, or dict.
    Falls back to numpy if torch is unavailable.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    # prefer torch
    if torch is not None:
        try:
            obj = torch.load(str(path), map_location="cpu")
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu()
            return obj
        except Exception:
            pass
    # numpy fallback (for .pt that are ndarrays)
    try:
        return np.load(str(path), allow_pickle=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")


# -----------------------------------------------------------------------------
# Pretty printing helpers
# -----------------------------------------------------------------------------


def _fmt_shape(x: Iterable[int]) -> str:
    try:
        return "x".join(str(int(v)) for v in x)
    except Exception:
        return str(tuple(x))


def summarize_array(name: str, arr: np.ndarray | "torch.Tensor") -> None:
    is_torch = torch is not None and isinstance(arr, torch.Tensor)
    if is_torch:
        shape = tuple(arr.shape)
        dtype = str(arr.dtype)
        # For big tensors, avoid full reduction on CPU if not necessary
        try:
            minv = float(arr.min().item())
            maxv = float(arr.max().item())
            meanv = (
                float(arr.float().mean().item())
                if arr.dtype != torch.uint8
                else float((arr.float()).mean().item())
            )
        except Exception:
            minv = maxv = meanv = float("nan")
    else:
        a = np.asarray(arr)
        shape = a.shape
        dtype = str(a.dtype)
        try:
            minv = float(np.min(a))
            maxv = float(np.max(a))
            meanv = float(np.mean(a))
        except Exception:
            minv = maxv = meanv = float("nan")

    print(f"   - shape: {_fmt_shape(shape)} | dtype: {dtype}")
    print(f"   - stats: min={minv:.3f} max={maxv:.3f} mean={meanv:.3f}")

    # Heuristics
    if len(shape) == 4 and shape[-1] in (1, 3):
        print(
            f"   - looks like video: {shape[0]} frames of {shape[1]}x{shape[2]}@{shape[3]}ch"
        )
    if len(shape) >= 2 and shape[-1] == 2:
        coord_type = "pixels" if maxv > 1.0 else "normalized [0,1]"
        print(f"   - looks like (x,y) coords in {coord_type}")


def _preview_element(elem: Any) -> str:
    """Make a short, informative preview string for an element."""
    try:
        if torch is not None and isinstance(elem, torch.Tensor):
            flat = elem.detach().cpu().reshape(-1)
            head = flat[:5].tolist()
            return f"Tensor shape={tuple(elem.shape)} dtype={elem.dtype} head={head}"
        if isinstance(elem, np.ndarray):
            flat = elem.reshape(-1)
            head = flat[:5].tolist()
            return f"ndarray shape={elem.shape} dtype={elem.dtype} head={head}"
        if isinstance(elem, dict):
            keys_preview = list(elem.keys())[:3]
            values_preview = []
            for k in keys_preview:
                v = elem[k]
                if isinstance(v, (int, float, str, bool)):
                    values_preview.append(f"{k}={v}")
                else:
                    values_preview.append(f"{k}={type(v).__name__}")
            return f"dict keys={list(elem.keys())[:5]}, sample_items=[{', '.join(values_preview)}]"
        if isinstance(elem, (list, tuple)):
            preview_items = []
            for i, item in enumerate(elem[:3]):
                if isinstance(item, (int, float, str, bool)):
                    preview_items.append(f"[{i}]={item}")
                else:
                    preview_items.append(f"[{i}]={type(item).__name__}")
            items_str = ", ".join(preview_items)
            return f"{type(elem).__name__} len={len(elem)}, items=[{items_str}]"
        # basic types
        s = repr(elem)
        if len(s) > 120:
            s = s[:120] + "..."
        return s
    except Exception:
        return f"{type(elem)}"


def summarize_list(name: str, lst: list[Any]) -> None:
    print(f"   - list length: {len(lst)}")
    if len(lst) == 0:
        return
    # Show first up to 5 elements with type and brief content
    limit = min(5, len(lst))
    for i in range(limit):
        elem = lst[i]
        print(f"     [{i}] type: {type(elem)} | {_preview_element(elem)}")

        # 为基本类型显示完整值
        if isinstance(elem, (int, float, str, bool, type(None))):
            print(f"         完整值: {elem}")
        elif isinstance(elem, (list, tuple)) and len(elem) <= 10:
            # 对于小的列表/元组，显示所有元素
            print(f"         完整内容: {elem}")
            # 如果元素是数值，尝试分析其含义
            if all(isinstance(x, (int, float)) for x in elem):
                if len(elem) == 2:
                    print(f"         -> 可能是(x, y)坐标: x={elem[0]}, y={elem[1]}")
                elif len(elem) == 3:
                    print(
                        f"         -> 可能是(x, y, z)坐标或RGB值: {elem[0]}, {elem[1]}, {elem[2]}"
                    )
        elif isinstance(elem, dict) and len(elem) <= 5:
            # 对于小的字典，显示所有键值对
            print(f"         完整内容: {elem}")
        elif isinstance(elem, (list, tuple)) and len(elem) > 10:
            # 对于大的列表，显示前几个和后几个元素
            preview = list(elem[:3]) + ["..."] + list(elem[-2:])
            print(f"         预览内容: {preview}")

    # If the first element is array-like, also summarize its stats/shape
    first = lst[0]
    if isinstance(first, (np.ndarray,)) or (
        torch is not None and isinstance(first, torch.Tensor)
    ):
        summarize_array(f"{name}[0]", first)
    elif isinstance(first, (list, tuple)):
        print(f"   - first item length: {len(first)}")
        # 如果第一个元素是小列表，显示其完整内容
        if len(first) <= 10:
            print(f"   - first item complete content: {first}")

    # 为gaze数据等特殊情况添加更多上下文
    if name.lower().find("gaze") != -1 and len(lst) > 0:
        print(f"   - 🔍 Gaze数据分析:")
        for i in range(min(3, len(lst))):
            elem = lst[i]
            if isinstance(elem, (list, tuple)) and len(elem) >= 2:
                print(f"     帧{i}: {elem} (可能表示注视点坐标)")
            elif isinstance(elem, tuple) and len(elem) == 3:
                print(f"     帧{i}: {elem} (可能表示3D注视点或包含置信度)")


def explore_pt_file(pt_path: Path) -> None:
    print(f"\n🔍 Exploring file: {pt_path}")
    print("-" * 60)
    try:
        obj = robust_load_pt(pt_path)
        print(f"Type: {type(obj)}")

        if torch is not None and isinstance(obj, torch.Tensor):
            summarize_array(pt_path.name, obj)
        elif isinstance(obj, np.ndarray):
            summarize_array(pt_path.name, obj)
        elif isinstance(obj, np.lib.npyio.NpzFile):
            keys = list(obj.files)
            print(f"NPZ keys ({len(keys)}): {keys}")
            for k in keys[:8]:
                try:
                    v = obj[k]
                    if isinstance(v, np.ndarray):
                        print(f"  • {k}: ndarray shape={v.shape} dtype={v.dtype}")
                    else:
                        print(f"  • {k}: {type(v)}")
                except Exception as e:
                    print(f"  • {k}: <error reading> {e}")
        elif isinstance(obj, dict):
            keys = list(obj.keys())
            print(f"Dict keys: {keys}")
            # Summarize up to a few entries
            for k in keys[:6]:
                v = obj[k]
                print(f"  • {k}: {type(v)}")

                # 为基本类型显示具体值
                if isinstance(v, (int, float, str, bool, type(None))):
                    print(f"    完整值: {v}")
                elif isinstance(v, (list, tuple)) and len(v) <= 10:
                    print(f"    完整内容: {v}")
                elif isinstance(v, dict) and len(v) <= 5:
                    print(f"    完整内容: {v}")
                elif isinstance(v, (np.ndarray,)) or (
                    torch is not None and isinstance(v, torch.Tensor)
                ):
                    summarize_array(k, v)
                elif isinstance(v, (list, tuple)):
                    summarize_list(k, list(v))
        elif isinstance(obj, (list, tuple)):
            summarize_list(pt_path.name, list(obj))
        else:
            print("(Unrecognized structure; printing repr head)")
            s = repr(obj)
            print(s[:300] + ("..." if len(s) > 300 else ""))
    except Exception as e:
        print(f"❌ Failed to load {pt_path}: {e}")


# -----------------------------------------------------------------------------
# Dataset directory helpers
# -----------------------------------------------------------------------------


def list_dataset_structure(data_path: Path) -> None:
    print(f"\n📂 Dataset directory: {data_path}")
    print("=" * 60)
    if not data_path.exists():
        print(f"❌ Dataset directory does not exist: {data_path}")
        return

    routes = sorted([d for d in data_path.glob("route_*") if d.is_dir()])
    if not routes:
        print("❌ No route directories found")
        return

    total_seeds = 0
    for route_dir in routes:
        seeds = sorted([d for d in route_dir.glob("seed_*") if d.is_dir()])
        total_seeds += len(seeds)
        print(f"  📁 {route_dir.name}: {len(seeds)} seeds")
        for i, seed_dir in enumerate(seeds[:3]):
            files = list(seed_dir.glob("*.pt"))
            json_files = list(seed_dir.glob("*.json"))
            print(
                f"     └─ {seed_dir.name}: {len(files)} .pt files, {len(json_files)} .json files"
            )
        if len(seeds) > 3:
            print(f"     └─ ... and {len(seeds) - 3} more seeds")
    print(f"\n📊 Total: {len(routes)} routes, {total_seeds} seeds")


def explore_seed_dir(seed_dir: Path) -> None:
    """Explore all relevant .pt files inside a seed directory."""
    print(f"\n📁 Inspecting seed: {seed_dir}")
    targets = [
        "observations.pt",
        "actions.pt",
        "gaze.pt",
        "gaze_pseudo.pt",
        "filter_dynamic.pt",
        "non_filter.pt",
    ]

    # list detected files
    available = sorted([p.name for p in seed_dir.glob("*.pt")])
    if available:
        print(f"Found .pt files ({len(available)}): {', '.join(available)}")
    else:
        print("No .pt files found.")

    # Explore targets first (if present), then any remaining .pt
    seen = set()
    for name in targets:
        p = seed_dir / name
        if p.exists():
            explore_pt_file(p)
            seen.add(name)

    for p in sorted(seed_dir.glob("*.pt")):
        if p.name in seen:
            continue
        explore_pt_file(p)

    # stats.json summary if present
    j = seed_dir / "stats.json"
    if j.exists():
        try:
            data = json.loads(j.read_text())
            if isinstance(data, dict):
                keys = list(data.keys())
                print(f"\n🧾 stats.json keys: {keys}")
        except Exception as e:
            print(f"(Failed to parse stats.json: {e})")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bench2Drive dataset structure exploration tool"
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="/path/to/dataset/bench2drive220",
        help="Dataset root directory",
    )
    p.add_argument("--route", type=str, default="route_2416", help="Route to explore")
    p.add_argument("--seed", type=str, default="seed_200", help="Seed to explore")
    p.add_argument(
        "--list-only",
        action="store_true",
        help="Only list structure; do not inspect files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_dir)

    print("🚀 Bench2Drive Dataset Exploration Tool")
    print("=" * 60)
    list_dataset_structure(data_path)

    if args.list_only:
        return

    seed_dir = data_path / args.route / args.seed
    if not seed_dir.exists():
        print(f"\n❌ Seed path does not exist: {seed_dir}")
        return
    explore_seed_dir(seed_dir)


if __name__ == "__main__":
    main()

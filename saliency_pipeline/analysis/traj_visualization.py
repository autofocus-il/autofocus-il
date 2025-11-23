#!/usr/bin/env python3
"""
Trajectory Visualization (per-seed GIF)

Generates a GIF for a specific route/seed by reading observations.pt.
All paths and parameters are controlled via YAML.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import yaml
import torch
import imageio.v2 as imageio

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # type: ignore

# Optional np/cv2 only when overlay is enabled

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "traj_visualization_config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_observations(obs_path: Path):
    data = torch.load(obs_path, map_location="cpu", weights_only=False)
    # minimal fallback tagging
    if isinstance(data, torch.Tensor):
        return data, "torch"
    return data, "other"


def to_uint8_rgb(frame: Any):
    # minimal fallback
    try:
        import numpy as np
    except Exception:
        raise RuntimeError("numpy required for visualization")
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    arr = frame
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


def process_one(cfg: Dict[str, Any]) -> Path:
    data_dir = Path(cfg["data"]["dataset_dir"]) / f"route_{cfg['seed']['route_id']}" / f"seed_{cfg['seed']['seed_id']}"
    obs_path = data_dir / "observations.pt"
    if not obs_path.exists():
        raise FileNotFoundError(f"observations.pt not found: {obs_path}")

    obs_data, _ = load_observations(obs_path)

    # iterate frames
    frame_step = int(cfg.get("processing", {}).get("frame_step", 1))
    max_frames = int(cfg.get("processing", {}).get("max_frames", 10**9))
    resize = cfg.get("processing", {}).get("resize", {})
    out_base = Path(cfg["output"]["out_dir"]) / f"route_{cfg['seed']['route_id']}"
    out_base.mkdir(parents=True, exist_ok=True)
    out_path = out_base / cfg["output"].get("gif_name", "{seed_id}_traj.gif").format(seed_id=cfg["seed"]["seed_id"])
    fps = int(cfg["output"].get("fps", 20))

    # get number of frames
    if isinstance(obs_data, torch.Tensor):
        N = int(obs_data.shape[0])
    else:
        try:
            import numpy as np
            N = int(np.asarray(obs_data).shape[0])
        except Exception:
            raise RuntimeError("Unsupported observations data type for visualization")

    frames_uint8: List[Any] = []
    iterator = range(0, min(N, max_frames), frame_step)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Collecting frames", unit="f")
    for i in iterator:
        frame_raw = obs_data[i] if isinstance(obs_data, torch.Tensor) else obs_data[i]
        img = to_uint8_rgb(frame_raw)
        # optional resize
        if resize:
            try:
                import cv2
                w = resize.get("width")
                h = resize.get("height")
                if w and h:
                    import numpy as np
                    img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass
        frames_uint8.append(img)

    if not frames_uint8:
        raise RuntimeError("No frames collected to write GIF")

    # GIF writing aligned with build_confunded_obs.py
    # We pass both fps and duration for broad compatibility across plugins
    try:
        imageio.mimsave(out_path, frames_uint8, fps=fps, duration=1.0 / max(1, fps))
    except TypeError:
        imageio.mimsave(out_path, frames_uint8, fps=fps)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-seed trajectory GIF (config-driven)",
    )
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "traj_visualization_config.yaml"), help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out = process_one(cfg)
    print(f"✅ Saved GIF: {out}")


if __name__ == "__main__":
    main()

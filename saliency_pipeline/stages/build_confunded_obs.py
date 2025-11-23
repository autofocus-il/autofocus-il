#!/usr/bin/env python3
"""
Build Confounded Observations (Brake + Steering Overlay)

This tool reads observations.pt for each frame and overlays simple driving
indicators derived from actions.pt:

- Brake indicator: a bright red dot at the bottom-left when braking
- Steering indicator: a white arrow (left/right) to the right of the dot

Run is driven by a YAML config under refactor/configs/, supporting:
- single_seed mode: also exports a GIF visualization for that seed
- all mode: writes processed frames back to observations.pt (overwrite)

The thresholds for interpreting actions reuse pipeline_utils.carla_action_to_text
and the shared thresholds under refactor/configs/pipeline_config.yaml.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
import torch
import imageio.v2 as imageio
import cv2
import numpy as np

# Local utilities
from saliency_pipeline.utils.pipeline_utils import load_pipeline_config, carla_action_to_text

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # type: ignore


# ----------------------- Config loading & routing -----------------------

def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "build_confunded_obs_config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_route(route_id: int) -> str:
    return f"route_{int(route_id)}"


def _to_seed(seed_id: int) -> str:
    return f"seed_{int(seed_id)}"


def enumerate_pairs(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return (route, seed) pairs based on cfg.run_mode.

    For mode == 'all', uses refactor/configs/pipeline_config.yaml routes_seeds.pairs
    to enumerate all routes and seeds.
    """
    run_cfg = cfg.get("run_mode", {})
    mode = (run_cfg.get("mode") or "single_seed").lower()

    if mode == "single_seed":
        ss = run_cfg.get("single_seed", {})
        return [(_to_route(ss["route_id"]), _to_seed(ss["seed_id"]))]
    elif mode == "single_route":
        sr = run_cfg.get("single_route", {})
        rid = int(sr["route_id"])  # required
        # try infer seeds from shared pipeline config; fallback 200..219
        try:
            _ = load_pipeline_config()  # ensure loaded
            from saliency_pipeline.utils.pipeline_utils import get_routes_seeds

            seeds = sorted({s for (r, s) in get_routes_seeds() if r == rid})
        except Exception:
            seeds = []
        if not seeds:
            seeds = list(range(200, 220))
        return [(_to_route(rid), _to_seed(s)) for s in seeds]
    elif mode == "all":
        from saliency_pipeline.utils.pipeline_utils import get_routes_seeds

        pairs = get_routes_seeds()
        return [(_to_route(r), _to_seed(s)) for (r, s) in pairs]
    else:
        raise ValueError(f"Invalid run_mode: {mode}")


# ----------------------- IO helpers -----------------------

def load_observations(obs_path: Path) -> Tuple[Union['np.ndarray', List['np.ndarray'], torch.Tensor], str]:
    """Load observations and return (data, type_tag).

    type_tag is one of: 'torch', 'np', 'list'. We preserve this upon saving.
    """
    data = torch.load(obs_path, map_location="cpu", weights_only=False)
    try:
        import numpy as np  # noqa: F401
    except Exception:
        np = None  # type: ignore
    else:
        np = __import__('numpy')
    if isinstance(data, torch.Tensor):
        return data, "torch"
    elif 'np' in globals() and np is not None and isinstance(data, np.ndarray):
        return data, "np"
    elif isinstance(data, (list, tuple)):
        # ensure list of frames
        return list(data), "list"
    else:
        # fallback: keep as-is; convert ad-hoc later
        return data, "other"


def load_actions(actions_path: Path) -> Optional['np.ndarray']:
    """Load actions.pt and return actions array of shape [N,7] if possible.
    Returns None if unavailable.
    """
    if not actions_path.exists():
        return None
    raw = torch.load(actions_path, map_location="cpu", weights_only=False)
    # Common formats: dict with 'actions' tensor; raw tensor; list
    if isinstance(raw, dict) and "actions" in raw:
        val = raw["actions"]
    else:
        val = raw
    try:
        import numpy as np
    except Exception:
        return None
    if isinstance(val, torch.Tensor):
        return val.cpu().numpy()
    elif isinstance(val, np.ndarray):
        return val
    elif isinstance(val, list):
        try:
            arr = np.asarray(val)
            return arr
        except Exception:
            return None
    return None


def save_observations(data: Union['np.ndarray', List['np.ndarray'], torch.Tensor, Any],
                      type_tag: str,
                      out_path: Path,
                      like_src: Any) -> None:
    """Save observations with the same structure as source when possible.

    - If source was torch.Tensor -> save torch.Tensor
    - If numpy array -> save numpy array
    - If list -> save list of numpy arrays
    Otherwise, fall back to torch.save(data).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if type_tag == "torch":
        if isinstance(data, torch.Tensor):
            torch.save(data, out_path)
        else:
            torch.save(torch.as_tensor(data), out_path)
    elif type_tag == "np":
        try:
            import numpy as np
        except Exception:
            torch.save(data, out_path)
        else:
            if isinstance(data, np.ndarray):
                torch.save(data, out_path)
            else:
                torch.save(np.asarray(data), out_path)
    elif type_tag == "list":
        if not isinstance(data, list):
            data = list(data)
        torch.save(data, out_path)
    else:
        torch.save(data, out_path)


# ----------------------- Rendering -----------------------

def _ensure_uint8_rgb(frame: Union['np.ndarray', torch.Tensor]) -> 'np.ndarray':
    """Convert a frame to uint8 RGB numpy array."""
    try:
        import numpy as np
    except Exception:
        raise RuntimeError("numpy is required for processing frames")
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    arr = frame
    # If channel-first, convert to HWC
    if hasattr(arr, 'ndim') and arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        # assume CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    # Normalize dtype
    if arr.dtype != np.uint8:
        # treat as float or other numeric scaled in [0,1]
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
    return arr


def _to_like_src_dtype(frame_uint8_rgb: 'np.ndarray', src_sample: Any) -> Any:
    """Convert uint8 RGB frame back to match dtype/range of src_sample."""
    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore
    # Detect destination characteristics
    if isinstance(src_sample, torch.Tensor):
        if torch.is_floating_point(src_sample):
            out = torch.from_numpy(frame_uint8_rgb.astype(np.float32) / 255.0) if np is not None else torch.from_numpy(frame_uint8_rgb)
        else:
            out = torch.from_numpy(frame_uint8_rgb.copy())
        return out
    else:
        if np is not None and isinstance(src_sample, np.ndarray):
            if np.issubdtype(src_sample.dtype, np.floating):
                return frame_uint8_rgb.astype(np.float32) / 255.0
            else:
                return frame_uint8_rgb.copy()
        else:
            # fallback to numpy uint8
            return frame_uint8_rgb.copy()


def draw_overlay(frame_rgb: 'np.ndarray', action: Optional['np.ndarray'], cfg: Dict[str, Any]) -> 'np.ndarray':
    """Draw brake dot and steering arrow on a single RGB frame.

    Args:
        frame_rgb: HxWx3 RGB uint8
        action: 7-dim CARLA action (throttle, steer, brake, handbrake, reverse, manual_gear, gear)
        cfg: configuration dictionary (render section)
    """
    h, w, _ = frame_rgb.shape
    render = cfg.get("render", {})
    dot_cfg = render.get("dot", {})
    arrow_cfg = render.get("arrow", {})

    # thresholds from shared pipeline config
    shared = load_pipeline_config()
    act_thresholds = shared.get("action_processing", {})
    brake_light = act_thresholds.get("braking", {}).get("light_threshold", 0.1)
    straight_thr = act_thresholds.get("steering", {}).get("straight_threshold", 0.05)
    throttle_light = act_thresholds.get("throttle", {}).get("light_threshold", 0.1)

    # Derive values
    brake_val = float(action[2]) if action is not None and len(action) > 2 else 0.0
    steer_val = float(action[1]) if action is not None and len(action) > 1 else 0.0
    throttle_val = float(action[0]) if action is not None and len(action) > 0 else 0.0

    # Work in BGR for OpenCV drawing and convert back to RGB
    out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Compute dot center by anchor
    radius = int(dot_cfg.get("radius", 7))
    # OpenCV uses BGR, we'll convert colors at draw-time
    red_bgr = tuple(int(c) for c in dot_cfg.get("color_bgr", [0, 0, 255]))

    anchor = render.get("anchor", "bottom_left").lower()
    if anchor == "top_mid":
        margin_top = int(dot_cfg.get("margin_top", 10))
        offset_x = int(dot_cfg.get("offset_x", 0))
        # Additional downward shift so upward arrow remains visible
        extra_down = int(render.get("top_mid_offset_y", 12))
        cx = max(radius, w // 2 + offset_x)
        cy = max(radius + margin_top + extra_down, radius)
    else:
        # default: bottom-left
        margin_left = int(dot_cfg.get("margin_left", 10))
        margin_bottom = int(dot_cfg.get("margin_bottom", 10))
        cx = max(radius + margin_left, radius)
        cy = h - max(radius + margin_bottom, radius)

    # Draw red dot only when braking above threshold
    if brake_val > brake_light and dot_cfg.get("enabled", True):
        cv2.circle(out, (cx, cy), radius, red_bgr, thickness=-1, lineType=cv2.LINE_AA)

    # Draw steering arrow for left/right when above straight threshold
    if arrow_cfg.get("enabled", True) and abs(steer_val) >= straight_thr:
        # Per-direction, YAML-configurable gaps and vertical offsets
        gap_default = int(arrow_cfg.get("gap_from_dot", 8))
        gap_left = int(arrow_cfg.get("gap_left", gap_default))
        gap_right = int(arrow_cfg.get("gap_right", gap_default))
        y_offset = int(arrow_cfg.get("y_offset", 0))
        base_length = int(arrow_cfg.get("length", 32))
        base_thickness = int(arrow_cfg.get("thickness", 2))
        base_head = int(arrow_cfg.get("head_size", 6))
        color_bgr = tuple(int(c) for c in arrow_cfg.get("color_bgr", [255, 255, 255]))

        y = int(np.clip(cy + y_offset, 0, h - 1)) if 'np' in globals() else cy + y_offset
        # Scale with steering magnitude beyond the straight threshold
        steer_mag = abs(steer_val)
        denom = max(1e-6, 1.0 - straight_thr)
        norm = min(1.0, max(0.0, (steer_mag - straight_thr) / denom))
        scale = 0.5 + 1.5 * norm  # 0.5x..2.0x
        length = max(6, int(base_length * scale))
        thickness = max(1, int(round(base_thickness * scale)))
        head_size = max(3, int(round(base_head * scale)))

        if steer_val < 0:
            # Left arrow on the LEFT side of the red dot, pointing LEFT
            end_x = max(0, cx - gap_left)
            start_x = max(0, end_x + length)
            start = (start_x, y)
            end = (end_x, y)
        else:
            # Right arrow on the RIGHT side of the red dot, pointing RIGHT
            start_x = min(w - 1, cx + gap_right)
            end_x = min(w - 1, start_x + length)
            start = (start_x, y)
            end = (end_x, y)
        cv2.arrowedLine(out, start, end, color_bgr, thickness=thickness, tipLength=max(0.1, head_size / max(length, 1)))

    # Draw straight acceleration arrow (upward) when nearly straight and throttle high
    if arrow_cfg.get("enabled", True) and abs(steer_val) < straight_thr and throttle_val > throttle_light:
        gap = int(arrow_cfg.get("gap_from_dot", 8))
        base_length = int(arrow_cfg.get("length", 32))
        base_thickness = int(arrow_cfg.get("thickness", 2))
        base_head = int(arrow_cfg.get("head_size", 6))
        color_bgr = tuple(int(c) for c in arrow_cfg.get("color_bgr", [255, 255, 255]))
        # Scale with throttle magnitude beyond light threshold
        denom = max(1e-6, 1.0 - throttle_light)
        norm = min(1.0, max(0.0, (throttle_val - throttle_light) / denom))
        scale = 0.5 + 1.5 * norm  # 0.5x..2.0x
        length = max(6, int(base_length * scale))
        thickness = max(1, int(round(base_thickness * scale)))
        head_size = max(3, int(round(base_head * scale)))
        x = cx
        start_y = max(0, cy - gap)
        end_y = max(0 + radius, start_y - length)  # keep within image top
        start = (x, start_y)
        end = (x, end_y)
        cv2.arrowedLine(out, start, end, color_bgr, thickness=thickness, tipLength=max(0.1, head_size / max(length, 1)))

    # Back to RGB for consistency with observations
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out_rgb


# ----------------------- Core processing -----------------------

def process_route_seed(route: str, seed: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_dir = Path(cfg["data"]["dataset_dir"]) / route / seed
    obs_path = data_dir / "observations.pt"
    actions_path = data_dir / "actions.pt"

    result = {
        "route": route,
        "seed": seed,
        "obs_path": str(obs_path),
        "gif_path": None,
        "written_pt": False,
        "frames": 0,
        "error": None,
    }

    if not obs_path.exists():
        result["error"] = f"Missing observations.pt: {obs_path}"
        return result

    try:
        obs_data, type_tag = load_observations(obs_path)
        # sample item to derive dtype/range
        if isinstance(obs_data, torch.Tensor):
            n_frames = int(obs_data.shape[0])
            sample = obs_data[0]
        else:
            try:
                import numpy as np
            except Exception:
                raise RuntimeError("numpy is required for processing observations")
            if isinstance(obs_data, np.ndarray):
                n_frames = int(obs_data.shape[0])
                sample = obs_data[0]
            elif isinstance(obs_data, list):
                n_frames = len(obs_data)
                sample = obs_data[0]
            else:
                # unknown type; try best-effort assuming array-like [T,H,W,3]
                arr = np.asarray(obs_data)
                n_frames = int(arr.shape[0])
                sample = arr[0]

        actions = load_actions(actions_path)
        # Align lengths
        if actions is not None:
            n_use = min(n_frames, actions.shape[0])
        else:
            n_use = n_frames

        # Iterate and render
        render_cfg = cfg
        frames_uint8: List['np.ndarray'] = []  # for GIF if needed

        def get_frame(i: int) -> Any:
            if isinstance(obs_data, torch.Tensor):
                return obs_data[i]
            else:
                try:
                    import numpy as np
                except Exception:
                    raise RuntimeError("numpy is required for processing observations")
                if isinstance(obs_data, np.ndarray):
                    return obs_data[i]
                elif isinstance(obs_data, list):
                    return obs_data[i]
                else:
                    return np.asarray(obs_data)[i]

        def set_frame(i: int, framed: Any) -> None:
            if isinstance(obs_data, torch.Tensor):
                obs_data[i] = framed
            else:
                try:
                    import numpy as np
                except Exception:
                    # if numpy missing, we can't meaningfully set into unknown structure
                    return
                if isinstance(obs_data, np.ndarray):
                    obs_data[i] = framed
                elif isinstance(obs_data, list):
                    obs_data[i] = framed
                else:
                    pass  # won't be used if type unknown

        for i in range(n_use):
            raw = get_frame(i)
            img_rgb = _ensure_uint8_rgb(raw)
            act = actions[i] if actions is not None and i < len(actions) else None
            over = draw_overlay(img_rgb, act, render_cfg)

            # Convert back to like-src dtype for writing back
            framed_like = _to_like_src_dtype(over, sample)
            set_frame(i, framed_like)

            # Save for GIF if single_seed
            frames_uint8.append(over)

        result["frames"] = n_use

        mode = (cfg.get("run_mode", {}).get("mode") or "single_seed").lower()
        if mode == "all" and cfg.get("processing", {}).get("write_back", True):
            save_observations(obs_data, type_tag, obs_path, like_src=sample)
            result["written_pt"] = True

        if mode == "single_seed":
            gifs_dir = Path(cfg.get("output", {}).get("gifs_dir", "confounded_gifs")) / route
            gifs_dir.mkdir(parents=True, exist_ok=True)
            fps = int(cfg.get("output", {}).get("fps", 20))
            gif_path = gifs_dir / f"{seed}_confounded.gif"
            imageio.mimsave(gif_path, frames_uint8, fps=fps)
            result["gif_path"] = str(gif_path)

        return result

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result


# ----------------------- Main -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Overlay brake/steering indicators onto observations.pt frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                                    # Use default YAML
  %(prog)s --config refactor/configs/build_confunded_obs_config.yaml
        """,
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "build_confunded_obs_config.yaml"),
        help="Path to YAML config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pairs = enumerate_pairs(cfg)
    mode = (cfg.get("run_mode", {}).get("mode") or "single_seed").lower()

    print("🚗 Build Confounded Observations")
    print("=" * 60)
    print(f"Targets: {len(pairs)} pairs")

    ok = 0
    iterator = pairs
    if mode == "all" and tqdm is not None:
        iterator = tqdm(pairs, desc="Processing seeds", unit="seed")

    for route, seed in iterator:
        if mode != "all":
            print(f"\n🎯 {route}/{seed}")
        res = process_route_seed(route, seed, cfg)
        if res.get("error"):
            if mode != "all":
                print(f"  ❌ {res['error']}")
        else:
            if mode != "all":
                if res.get("gif_path"):
                    print(f"  🎬 GIF: {res['gif_path']}")
                if res.get("written_pt"):
                    print(f"  💾 Overwrote: {res['obs_path']}")
                print(f"  ✅ Frames processed: {res['frames']}")
            ok += 1

    print(f"\nDone. Success: {ok}/{len(pairs)}")


if __name__ == "__main__":
    main()

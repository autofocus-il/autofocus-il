#!/usr/bin/env python3
"""
Global Description Generator (YAML-configured)

Generates global scene descriptions for trajectories using a VLM.
Supports two dataset layouts selectable via YAML:
- bench2drive: reads `observations.pt` under `{route}/{seed}`
- bdv2: reads raw JPG frames under
  `{task}/{timestamp}/raw/{traj_group}/{traj}/images{camera}/im_*.jpg`

Stages:
1) Load observations from dataset
2) Evenly sample k frames
3) Convert frames to base64 (configurable format)
4) Query VLM with multi-image prompt
5) Save structured JSON to configured output path

Config: refactor/configs/global_desc_config.yaml

Usage:
    python refactor/get_global_desc.py
"""

import os
import io
import json
import time
from typing import Dict, Any, List, Tuple, Optional

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import yaml

from saliency_pipeline.utils.pipeline_utils import (
    ROUTES_SEEDS,  # list of (route_id, seed_id)
    get_api_client,
    load_pipeline_config,
    image_to_base64,
)
from saliency_pipeline.utils.config_manager import load_merged_config
import argparse

# -----------------------------
# Config loading
# -----------------------------

def load_global_desc_config(config_path: Optional[Path] = None, domain: Optional[str] = None) -> Dict[str, Any]:
    """Load config using two-domain layout (merging common + domain)."""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "global_desc_config.yaml"
    return load_merged_config(Path(config_path), domain=domain)

class GlobalDescGenerator:
    """OOP pipeline for generating global scene descriptions."""

    def __init__(self, gd_config: Dict[str, Any], pipeline_config: Dict[str, Any],
                 client=None) -> None:
        self.gd_config = gd_config
        self.pipeline_config = pipeline_config
        self.client = client or get_api_client()

        # Cached frequently used fields
        ds_cfg = gd_config.get("dataset", {})
        self.dataset_type: str = ds_cfg.get("type", "bench2drive").lower()
        self.data_dir: str = gd_config["data"].get("data_dir", "")
        self.bdv2_cfg: Dict[str, Any] = gd_config["data"].get("bdv2", {})
        self.k_frames: int = gd_config["data"]["processing"]["k_frames"]
        self.rng_seed: int = gd_config["data"]["processing"].get("seed", 42)
        self.prompt: str = gd_config["vlm"]["prompt_template"]
        self.model: str = gd_config["vlm"]["model"]
        # Optional fallbacks to handle different API deployments
        self.fallback_models: List[str] = gd_config.get("vlm", {}).get(
            "fallback_models",
            [
                "Qwen/Qwen2.5-VL-72B-Instruct",
                "Qwen/Qwen2.5-VL-32B-Instruct",
                "Qwen/Qwen2.5-VL-7B-Instruct",
            ],
        )
        self.api_provider: str = gd_config.get("vlm", {}).get("api_provider", "OpenAI")
        self.shared_api: Dict[str, Any] = gd_config.get("shared_api", {})
        self.img_format: str = pipeline_config["processing"]["image"]["format"]
        api_cfg = gd_config.get("api", {})
        self.timeout_s: int = api_cfg.get("timeout_seconds", pipeline_config["processing"]["vlm"].get("timeout_seconds", 60))
        self.max_retries: int = api_cfg.get("max_retries", pipeline_config["processing"]["vlm"].get("max_retries", 3))
        out_cfg = gd_config["output"]
        self.base_output_dir: Path = Path(out_cfg["base_output_dir"]).resolve()
        self.result_filename: str = out_cfg.get("result_filename", "result.json")
        jf = out_cfg.get("json_formatting", {})
        self.json_indent: int = jf.get("indent", 2)
        self.json_ensure_ascii: bool = jf.get("ensure_ascii", False)

        np.random.seed(self.rng_seed)

    # ===== Stages as methods =====
    # ===== Bench2Drive loader =====
    def _load_data(self, route: str, seed: str) -> np.ndarray:
        obs_path = Path(self.data_dir) / route / seed / "observations.pt"
        obs_data = torch.load(obs_path, map_location="cpu", weights_only=False)
        if isinstance(obs_data, torch.Tensor):
            obs_data = obs_data.numpy()
        assert obs_data.ndim == 4 and obs_data.shape[-1] == 3
        return obs_data

    # ===== BDV2 helpers =====
    @staticmethod
    def _frame_index_from_name(name: str) -> int:
        base = Path(name).stem
        try:
            return int(base.split("_")[-1])
        except Exception:
            return -1

    def _load_bdv2_frames(self, task: str, timestamp: str, traj_group: str, traj: str, camera: int,
                           frame_step: int = 1, frame_glob: str = "im_*.jpg") -> np.ndarray:
        root = Path(self.bdv2_cfg.get("dataset_root", "/path/to/dataset/bdv2"))
        images_dir = root / task / timestamp / "raw" / traj_group / traj / f"images{camera}"
        frames = list(images_dir.glob(frame_glob))
        frames.sort(key=lambda p: (self._frame_index_from_name(p.name), p.name))
        frames = frames[:: max(1, int(frame_step))]
        arrs: List[np.ndarray] = []
        for p in frames:
            try:
                im = Image.open(p).convert("RGB")
                arrs.append(np.array(im))
            except Exception:
                continue
        if not arrs:
            raise FileNotFoundError(f"No frames found under {images_dir}")
        return np.stack(arrs, axis=0)

    def _sample_frames(self, frames: np.ndarray) -> np.ndarray:
        total_frames = len(frames)
        indices = np.linspace(0, total_frames - 1, self.k_frames, dtype=int)
        sampled = frames[indices]
        assert len(sampled) == self.k_frames
        assert sampled.shape[1:] == frames.shape[1:]
        return sampled

    def _frames_to_base64(self, frames: np.ndarray) -> List[str]:
        out: List[str] = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            img = Image.fromarray(frame)
            out.append(image_to_base64(img, format_override=self.img_format))
        return out

    def _query_vlm(self, base64_images: List[str]) -> Dict[str, Any]:
        img_format_lower = self.img_format.lower()
        content = []
        for b64 in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{img_format_lower};base64,{b64}",
                    "detail": "low"
                }
            })
        content.append({"type": "text", "text": self.prompt})
        
        # Adjust system message based on dataset type
        if self.dataset_type == "bdv2":
            system_msg = "You are an expert in household manipulation scene analysis. Always return valid JSON."
        else:
            system_msg = "You are an expert in autonomous driving scene analysis. Always return valid JSON."
            
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_msg}]},
            {"role": "user", "content": content}
        ]

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # Try primary model then fallbacks if model-not-found
                candidate_models = [self.model] + [m for m in self.fallback_models if m != self.model]
                last_model_err: Optional[Exception] = None
                response_text: Optional[str] = None
                for mdl in candidate_models:
                    try:
                        completion = self.client.chat.completions.create(
                            model=mdl,
                            messages=messages,
                        )
                        response_text = completion.choices[0].message.content
                        if not isinstance(response_text, str) or not response_text:
                            raise ValueError("Empty VLM response")
                        # Success
                        break
                    except Exception as me:
                        msg = str(me)
                        if "Model does not exist" in msg or "20012" in msg:
                            last_model_err = me
                            continue
                        # Other errors should bubble up to outer retry
                        raise
                if response_text is None:
                    raise last_model_err or RuntimeError("All candidate VLM models failed")
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    import re
                    m = re.search(r"\{.*\}", response_text, re.DOTALL)
                    if m:
                        try:
                            return json.loads(m.group())
                        except json.JSONDecodeError:
                            pass
                    return {"objects": [], "description": response_text}
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(min(2 * attempt, 5))
                    continue
                raise e

    def _save_result(self, vlm_result: Dict[str, Any], route: str, seed: str) -> Dict[str, Any]:
        route_dir = self.base_output_dir / route
        route_dir.mkdir(parents=True, exist_ok=True)
        output_file = route_dir / self.result_filename

        final = {
            "route": route,
            "seed": seed,
            "k_frames": self.k_frames,
            "objects": vlm_result.get("objects", []),
            "description": vlm_result.get("description", ""),
            "metadata": {
                "total_objects": len(vlm_result.get("objects", [])),
                "object_types": list(set(obj.get("type", "unknown") for obj in vlm_result.get("objects", []))),
                "processed_frames": self.k_frames,
            },
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=self.json_indent, ensure_ascii=self.json_ensure_ascii)
        print(f"Saved VLM analysis to {output_file}")
        return {
            "saved_to": str(output_file),
            "total_objects": len(final["objects"]),
            "description_length": len(final["description"]),
        }

    # ===== Public API =====
    def run_single(self, route: str, seed_name: str) -> Dict[str, Any]:
        if self.dataset_type == "bench2drive":
            frames = self._load_data(route, seed_name)
            sampled_frames = self._sample_frames(frames)
            b64_images = self._frames_to_base64(sampled_frames)
            vlm_json = self._query_vlm(b64_images)
            try:
                objs = vlm_json.get("objects", []) if isinstance(vlm_json, dict) else []
            except Exception:
                objs = []
            if not objs:
                # If parsed objects list is empty, retry once
                print("[GlobalDesc] Empty objects from VLM; retrying once...")
                vlm_json = self._query_vlm(b64_images)
            result_summary = self._save_result(vlm_json, route, seed_name)
            return {
                "total_frames": len(frames),
                "sampled_frames": len(sampled_frames),
                "route_seed": f"{route}/{seed_name}",
                "objects_found": result_summary["total_objects"],
                "description_length": result_summary["description_length"],
                "saved_to": result_summary["saved_to"],
            }
        else:
            raise RuntimeError("Use run_single_bdv2 for bdv2 dataset type")

    def run_single_bdv2(self, task: str, timestamp: str, traj_group: str, traj_name: str, camera: int,
                         frame_step: int = 1, frame_glob: str = "im_*.jpg") -> Dict[str, Any]:
        frames = self._load_bdv2_frames(task, timestamp, traj_group, traj_name, camera, frame_step, frame_glob)
        sampled_frames = self._sample_frames(frames)
        b64_images = self._frames_to_base64(sampled_frames)
        vlm_json = self._query_vlm(b64_images)
        try:
            objs = vlm_json.get("objects", []) if isinstance(vlm_json, dict) else []
        except Exception:
            objs = []
        if not objs:
            print("[GlobalDesc] Empty objects from VLM (BDV2); retrying once...")
            vlm_json = self._query_vlm(b64_images)
        # Route/seed labels for saving（任务级保存到 bdv2/{task}/result.json）
        route = f"bdv2/{task}"
        seed_name = f"images{camera}"
        result_summary = self._save_result(vlm_json, route, seed_name)
        return {
            "total_frames": len(frames),
            "sampled_frames": len(sampled_frames),
            "route_seed": f"{route}/{seed_name}",
            "objects_found": result_summary["total_objects"],
            "description_length": result_summary["description_length"],
            "saved_to": result_summary["saved_to"],
        }

    def run(self) -> None:
        mode = self.gd_config["run_mode"]["mode"]
        if self.dataset_type == "bench2drive":
            if mode == "single_run":
                route = self.gd_config["run_mode"]["single_run"]["route"]
                seed_name = self.gd_config["run_mode"]["single_run"]["seed_name"]
                print(f"[Single] Running {route}/{seed_name}")
                summary = self.run_single(route, seed_name)
                print(summary)
            elif mode == "batch":
                print("[Batch] Running all routes with seed_200 from ROUTES_SEEDS")
                all_summaries: List[Dict[str, Any]] = []
                success = 0
                fail = 0
                filtered_pairs = [(route_id, seed_id) for (route_id, seed_id) in ROUTES_SEEDS if seed_id == 200]
                total = len(filtered_pairs)
                for idx, (route_id, seed_id) in enumerate(filtered_pairs, 1):
                    route = f"route_{route_id}"
                    seed_name = f"seed_{seed_id}"
                    print(f"[{idx}/{total}] {route}/{seed_name}")
                    try:
                        summary = self.run_single(route, seed_name)
                        all_summaries.append(summary)
                        success += 1
                    except Exception as e:
                        print(f"  Error: {e}")
                        all_summaries.append({"route_seed": f"{route}/{seed_name}", "error": str(e)})
                        fail += 1
                print({
                    "total": total,
                    "success": success,
                    "failed": fail,
                    "samples": all_summaries[:5],
                })
            else:
                raise ValueError(f"Invalid run mode: {mode}")
        elif self.dataset_type == "bdv2":
            if mode == "single_traj":
                sr = self.gd_config["run_mode"]["single_traj"]
                summary = self.run_single_bdv2(
                    sr["task"], sr["timestamp"], sr["traj_group"], sr["traj_name"], int(sr.get("camera", 0)),
                    self.bdv2_cfg.get("frame_step", 1), self.bdv2_cfg.get("frame_glob", "im_*.jpg"),
                )
                print(summary)
            elif mode == "batch_task":
                bt = self.gd_config["run_mode"]["batch_task"]
                task = bt["task"]
                camera = int(bt.get("camera", 0))
                # 仅选择一个默认轨迹（优先 timestamp 最早、traj_group0/traj0），跑一次
                root = Path(self.bdv2_cfg.get("dataset_root", "/path/to/dataset/bdv2"))
                sel_ts = None
                for ts_dir in sorted((root / task).iterdir()):
                    raw = ts_dir / "raw"
                    if not raw.exists():
                        continue
                    pref = raw / "traj_group0" / "traj0"
                    if pref.exists():
                        sel_ts = (ts_dir.name, "traj_group0", "traj0")
                        break
                    # fallback
                    g_list = sorted(d for d in raw.iterdir() if d.is_dir() and d.name.startswith("traj_group"))
                    if g_list:
                        t_list = sorted(d for d in g_list[0].iterdir() if d.is_dir() and d.name.startswith("traj"))
                        if t_list:
                            sel_ts = (ts_dir.name, g_list[0].name, t_list[0].name)
                            break
                if not sel_ts:
                    raise FileNotFoundError(f"No valid trajectory found under task {task}")
                ts, g, tr = sel_ts
                summary = self.run_single_bdv2(task, ts, g, tr, camera,
                                               self.bdv2_cfg.get("frame_step", 1), self.bdv2_cfg.get("frame_glob", "im_*.jpg"))
                print(summary)
            elif mode == "batch_all":
                # 对每个 task 仅跑一次（默认轨迹）
                ba = self.gd_config["run_mode"].get("batch_all", {})
                camera = int(ba.get("camera", 0))
                root = Path(self.bdv2_cfg.get("dataset_root", "/path/to/dataset/bdv2"))
                total = 0
                for task_dir in sorted(d for d in root.iterdir() if d.is_dir()):
                    task = task_dir.name
                    # 复用上面的选择逻辑
                    sel_ts = None
                    for ts_dir in sorted((root / task).iterdir()):
                        raw = ts_dir / "raw"
                        if not raw.exists():
                            continue
                        pref = raw / "traj_group0" / "traj0"
                        if pref.exists():
                            sel_ts = (ts_dir.name, "traj_group0", "traj0")
                            break
                        g_list = sorted(d for d in raw.iterdir() if d.is_dir() and d.name.startswith("traj_group"))
                        if g_list:
                            t_list = sorted(d for d in g_list[0].iterdir() if d.is_dir() and d.name.startswith("traj"))
                            if t_list:
                                sel_ts = (ts_dir.name, g_list[0].name, t_list[0].name)
                                break
                    if not sel_ts:
                        continue
                    ts, g, tr = sel_ts
                    summary = self.run_single_bdv2(task, ts, g, tr, camera,
                                                   self.bdv2_cfg.get("frame_step", 1), self.bdv2_cfg.get("frame_glob", "im_*.jpg"))
                    print(summary)
                    total += 1
                print({"processed_tasks": total, "camera": camera})
            else:
                raise ValueError(f"Invalid run mode for bdv2: {mode}")
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate global scene descriptions (two-domain YAML)")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "global_desc_config.yaml"),
                        help="Path to YAML config (supports bench2drive|bdv2 domains)")
    parser.add_argument("--domain", choices=["bench2drive", "bdv2"], default=None,
                        help="Override active domain in config (optional)")
    args = parser.parse_args()

    # Load configs
    gd_config = load_global_desc_config(Path(args.config), domain=args.domain)
    pipeline_config = load_pipeline_config()
    pipeline = GlobalDescGenerator(gd_config, pipeline_config)
    pipeline.run()

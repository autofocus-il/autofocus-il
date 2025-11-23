#!/usr/bin/env python3
"""
VLM-Filtered Object Detection Pipeline for Autonomous Driving (Refactored OOP Version)

This module implements a VLM (Vision Language Model) filtered object detection pipeline
that uses Grounding DINO for initial detection, applies object tracking, and then uses
VLM to intelligently filter important objects based on driving context and intentions.

Changes in this refactor:
- Converted to an object-oriented design (VLMFilterPipeline) instead of function-style
- Inlined stages 5–7 previously referenced from vlm_fliter.py (no external dependency)
- Added multi-threaded execution where each route runs in its own thread and GPU
- Preserves YAML-based configuration semantics
"""

import os
import sys
import json
import pickle
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import threading
import requests

import numpy as np
import torch
import cv2
import imageio.v2 as imageio
from PIL import Image
try:
    from ultralytics import YOLO as UltralyticsYOLO
except Exception:
    UltralyticsYOLO = None
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import supervision as sv
from supervision.draw.color import ColorPalette

# Import VLM utilities
from saliency_pipeline.utils.pipeline_utils import (
    calc_IoU, grounding_filter, normalize_bbox,
    carla_action_to_text, extract_global_intent, extract_action_intent,
    build_vlm_prompt_topk_bdv2, build_vlm_prompt_topk_bench2drive, build_vlm_prompt_topk,
    parse_vlm_topk_response, query_vlm, image_to_base64,
    simple_object_tracker, is_trigger_by_id, single_frame_redetect,
    get_routes_seeds, load_pipeline_config
)
from saliency_pipeline.utils.config_manager import load_merged_config
import argparse


def load_vlm_filter_config(config_path: Optional[Path] = None, domain: Optional[str] = None) -> Dict[str, Any]:
    """Load VLM filter configuration using two-domain layout (merging common + domain).

    Falls back to legacy layout if provided.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "vlm_filter_config.yaml"
    
    # For generated configs, load directly without domain merging
    if "_generated" in str(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = load_merged_config(Path(config_path), domain=domain)
    # Backward-compatible filename defaults for existing_json
    det = cfg.setdefault("detection", {})
    ex = det.setdefault("existing_json", {})
    if not ex.get("filename"):
        default_gd_name = cfg.get("output", {}).get("filenames", {}).get("grounding_detections", "grounding_detections.json")
        ex["filename"] = default_gd_name
    # API defaults
    api = cfg.setdefault("api", {})
    api.setdefault("urls", [])
    api.setdefault("urls_file", str((Path(__file__).parent.parent / "grounding_api" / "api_urls.txt").resolve()))
    api.setdefault("timeout", 30)
    api.setdefault("num_workers", 8)
    # BDV2 defaults
    bd = cfg.setdefault("bdv2", {})
    bd.setdefault("dataset_root", "/path/to/dataset/bdv2")
    bd.setdefault("frame_glob", "im_*.jpg")
    bd.setdefault("frame_step", 1)
    bd.setdefault("camera", 0)
    return cfg

def get_run_parameters(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Extract run mode and parameters from configuration."""
    run_config = config["run_mode"]
    mode = run_config["mode"]
    
    if mode == "single_seed":
        if "single_seed" not in run_config:
            raise ValueError("single_seed configuration not found")
        return mode, run_config["single_seed"]
    elif mode == "single_route":
        if "single_route" not in run_config:
            raise ValueError("single_route configuration not found")
        return mode, run_config["single_route"]
    elif mode == "all":
        return mode, {}
    else:
        raise ValueError(f"Invalid run mode: {mode}")

class VLMFilterPipeline:
    """OOP pipeline encapsulating model, config, and all processing stages."""

    def __init__(self, config: Dict[str, Any], device_override: Optional[str] = None,
                 shared_processor: Optional[Any] = None,
                 shared_model: Optional[Any] = None,
                 inference_lock: Optional[threading.Lock] = None):
        self.config = config
        # Resolve device: thread may override per-GPU
        self.device = device_override or config["model"].get("device", "cpu")
        self.backend = self.config.get("detection", {}).get("backend", "local").lower()
        # Visualization helpers
        self.palette = ColorPalette.from_hex(config.get("visualization", {})
                                             .get("color_palette", [
                                                 "#e6194b", "#3cb44b", "#ffe119", "#0082c8",
                                                 "#f58231", "#911eb4", "#46f0f0", "#f032e6",
                                                 "#d2f53c", "#fabebe", "#008080", "#e6beff",
                                                 "#aa6e28", "#fffac8", "#800000", "#aaffc3"
                                             ]))
        self.model = shared_model
        self.processor = shared_processor
        # YOLO (optional) for gripper detection
        yolo_cfg = self.config.get("yolo", {})
        self.yolo_enabled = bool(yolo_cfg.get("enabled", False))
        self.yolo_model_path = yolo_cfg.get("model_path")
        self.yolo_conf_threshold = float(yolo_cfg.get("conf_threshold", 0.85))
        self.yolo_class_map = yolo_cfg.get("class_map", {0: "gripper"})
        self._yolo_model = None
        if self.yolo_enabled and UltralyticsYOLO is None:
            print("[WARN] YOLO enabled but ultralytics not available. Disabling YOLO stage.")
            self.yolo_enabled = False
        self.forward_lock = inference_lock
        # API endpoints (if using API backend)
        self.api_urls = self._load_api_urls()
        self._active_api_url: Optional[str] = None
        self.dataset_type: str = config.get("dataset", {}).get("type", "bench2drive").lower()
        self.bdv2_cfg: Dict[str, Any] = config.get("bdv2", {})

    # ===== Stage 1: Load model =====
    def load_model(self):
        if self.backend == "api":
            # No local model required when using API backend
            return
        model_id = self.config["model"]["model_id"]
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(model_id)
        if self.model is None:
            # Load fully on CPU, then move to target device explicitly to avoid meta tensors
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id,
                low_cpu_mem_usage=False,
                device_map=None
            )
            if self.device.startswith("cuda"):
                # Move parameters to device
                self.model = self.model.to(self.device)
            self.model.eval()
        print(f"Model loaded on {self.device}")

    # ===== Stage 2: Load data (Bench2Drive) =====
    def load_data(self, route: str, seed: str) -> Optional[Dict[str, Any]]:
        data_dir = self.config["data"]["data_dir"]
        base_path = Path(data_dir) / route / seed
        obs_path = base_path / "observations.pt"
        if not obs_path.exists():
            print(f"File not found: {obs_path}")
            return None
        obs_data = torch.load(obs_path, map_location="cpu", weights_only=False)
        if isinstance(obs_data, torch.Tensor):
            obs_data = obs_data.numpy()
        actions_path = base_path / "actions.pt"
        actions_data = None
        if actions_path.exists():
            actions_raw = torch.load(actions_path, map_location="cpu", weights_only=False)
            if isinstance(actions_raw, dict) and "actions" in actions_raw:
                actions_data = actions_raw["actions"]
        stats_path = base_path / "stats.json"
        stats_data = None
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats_data = json.load(f)
        print(f"Loaded {len(obs_data)} frames, {len(actions_data) if actions_data is not None else 0} actions")
        return {
            "frames": obs_data,
            "actions": actions_data,
            "stats": stats_data,
            "route": route,
            "seed": seed,
            "num_frames": len(obs_data),
        }

    # ===== Stage 2b: Load data (BDV2) =====
    @staticmethod
    def _frame_index_from_name(name: str) -> int:
        base = Path(name).stem
        try:
            return int(base.split("_")[-1])
        except Exception:
            return -1

    # Safe pickle loader (avoid ROS dependency issues)
    class _Dummy:  # noqa: D401 simple placeholder
        def __init__(self, *a, **k):
            pass

    class _SafeUnpickler(pickle.Unpickler):  # type: ignore[name-defined]
        def find_class(self, module, name):  # pragma: no cover
            try:
                __import__(module)
                mod = sys.modules[module]
                return getattr(mod, name)
            except Exception:
                return type(name, (VLMFilterPipeline._Dummy,), {})

    @staticmethod
    def _safe_pickle_load(path: Path):  # pragma: no cover
        with open(path, "rb") as f:
            return VLMFilterPipeline._SafeUnpickler(f).load()

    @staticmethod
    def _stack_actions_from_policy_list(pol: Any) -> Optional[np.ndarray]:
        if not isinstance(pol, list) or not pol:
            return None
        rows: List[np.ndarray] = []
        for step in pol:
            if isinstance(step, dict):
                a = step.get("actions")
                if a is None:
                    continue
                if isinstance(a, np.ndarray):
                    rows.append(a.astype(float))
                elif isinstance(a, (list, tuple)):
                    rows.append(np.asarray(a, dtype=float))
        if not rows:
            return None
        max_d = max(r.shape[-1] for r in rows)
        out = []
        for r in rows:
            if r.shape[-1] == max_d:
                out.append(r)
            else:
                tmp = np.zeros((max_d,), dtype=float)
                tmp[: r.shape[-1]] = r
                out.append(tmp)
        return np.stack(out, axis=0)

    def load_data_bdv2(self, task: str, timestamp: str, traj_group: str, traj_name: str, camera: int) -> Optional[Dict[str, Any]]:
        root = Path(self.bdv2_cfg.get("dataset_root", "/path/to/dataset/bdv2"))
        images_dir = root / task / timestamp / "raw" / traj_group / traj_name / f"images{camera}"
        if not images_dir.exists():
            print(f"[BDV2] Missing images dir: {images_dir}")
            return None
        frames = list(images_dir.glob(self.bdv2_cfg.get("frame_glob", "im_*.jpg")))
        frames.sort(key=lambda p: (self._frame_index_from_name(p.name), p.name))
        frames = frames[:: max(1, int(self.bdv2_cfg.get("frame_step", 1)))]
        imgs: List[np.ndarray] = []
        for p in frames:
            try:
                im = Image.open(p).convert("RGB")
                imgs.append(np.array(im))
            except Exception:
                continue
        if not imgs:
            print(f"[BDV2] No frames found under {images_dir}")
            return None
        # Try to load actions from policy_out.pkl
        actions_arr = None
        pol_path = images_dir.parent / "policy_out.pkl"
        if pol_path.exists():
            try:
                pol = self._safe_pickle_load(pol_path)
                actions_arr = self._stack_actions_from_policy_list(pol)
            except Exception:
                actions_arr = None
        data = {
            "frames": np.stack(imgs, axis=0),
            "actions": actions_arr,
            "stats": None,
            "route": f"bdv2/{task}/{timestamp}/{traj_group}/{traj_name}",
            "seed": f"images{camera}",
            "num_frames": len(imgs),
        }
        print(f"[BDV2] Loaded {data['num_frames']} frames, actions: {None if actions_arr is None else actions_arr.shape}")
        return data

    # ===== Stage 3: Detection =====
    def detect_objects(self, data: Dict[str, Any], api_url: Optional[str] = None) -> Dict[str, Any]:
        frames = data["frames"]
        processing_config = self.config["data"]["processing"]
        detection_config = self.config["detection"]
        all_detections: List[Dict[str, Any]] = []
        max_frames = min(processing_config["max_frames"], len(frames))
        text_prompt = detection_config["text_prompt"]
        print(f"Processing {max_frames} frames with prompt: {text_prompt}")
        if self.backend == "api":
            effective_api_url = api_url or self._active_api_url or (self.api_urls[0] if self.api_urls else None)
            if not effective_api_url:
                raise RuntimeError("API backend selected but no API URL available. Please launch servers and/or configure api.urls or api.urls_file.")
            return self._detect_objects_via_api(data, effective_api_url)
        
        # Ensure model is loaded for local backend
        if self.model is None or self.processor is None:
            self.load_model()
        for frame_idx in range(0, max_frames, processing_config["frame_step"]):
            frame = frames[frame_idx]
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            image = Image.fromarray(frame)
            inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
            if self.device.startswith("cuda"):
                inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            # Forward pass (thread-safe if shared)
            with torch.no_grad():
                if self.forward_lock is not None:
                    with self.forward_lock:
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results_processed = self.processor.post_process_grounded_object_detection(
                outputs, input_ids=inputs["input_ids"], target_sizes=target_sizes
            )[0]
            # Grounding detections
            detections_grounding = []
            if "scores" in results_processed and len(results_processed["scores"]) > 0:
                boxes = results_processed["boxes"].cpu().numpy()
                scores = results_processed["scores"].cpu().numpy()
                labels = results_processed["labels"]
                for box, score, label in zip(boxes, scores, labels):
                    norm_box = normalize_bbox(box.tolist(), frame.shape[:2])
                    detections_grounding.append({"label": label, "bbox": norm_box, "score": float(score)})

            # YOLO gripper detections (optional) -> normalize bbox and merge into detections
            yolo_detections: List[Dict[str, Any]] = []
            if self.yolo_enabled:
                if self._yolo_model is None:
                    try:
                        if not self.yolo_model_path:
                            raise RuntimeError("YOLO model_path is empty")
                        self._yolo_model = UltralyticsYOLO(self.yolo_model_path)
                        print(f"[YOLO] Loaded model from {self.yolo_model_path}")
                    except Exception as e:
                        print(f"[YOLO ERROR] Failed to load model: {e}")
                        self.yolo_enabled = False
                if self.yolo_enabled and self._yolo_model is not None:
                    try:
                        # Ultralytics accepts numpy arrays in RGB
                        np_img = np.array(image)
                        yolo_results = self._yolo_model.predict(source=np_img, verbose=False)
                        if yolo_results:
                            r0 = yolo_results[0]
                            if r0.boxes is not None and len(r0.boxes) > 0:
                                h, w = np_img.shape[:2]
                                for i in range(len(r0.boxes)):
                                    conf = float(r0.boxes.conf[i].item())
                                    if conf < self.yolo_conf_threshold:
                                        continue
                                    xyxy = r0.boxes.xyxy[i].tolist()
                                    cls_id = int(r0.boxes.cls[i].item()) if getattr(r0.boxes, 'cls', None) is not None else 0
                                    x1, y1, x2, y2 = xyxy
                                    cx = ((x1 + x2) / 2.0) / float(w)
                                    cy = ((y1 + y2) / 2.0) / float(h)
                                    # Also keep full bbox normalized
                                    nx1, ny1, nx2, ny2 = x1/float(w), y1/float(h), x2/float(w), y2/float(h)
                                    label_name = self.yolo_class_map.get(cls_id, "gripper")
                                    yolo_detections.append({
                                        "label": label_name,
                                        "bbox": [float(nx1), float(ny1), float(nx2), float(ny2)],
                                        "score": conf,
                                    })
                    except Exception as e:
                        print(f"[YOLO ERROR] Inference failed on frame {frame_idx}: {e}")

            # Merge grounding + yolo detections and apply score threshold filter once
            merged_detections = detections_grounding + yolo_detections
            detections = grounding_filter(merged_detections, detection_config["box_threshold"])

            all_detections.append({
                "frame_idx": frame_idx,
                "detections": detections,
                "image": image,
            })
            if frame_idx % 50 == 0:
                print(f"Frame {frame_idx}: {len(detections)} objects detected")
        return {
            "route": data["route"],
            "seed": data["seed"],
            "frame_detections": all_detections,
            "frames_processed": len(all_detections),
        }

    # --- Load detections from precomputed JSON ---
    def load_detections_from_json(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load precomputed grounding detections for a route/seed and attach images.

        Looks up JSON under either `detection.existing_json.base_dir` (if set) or
        `output.base_output_dir`, then `{route}/{seed}/{filename}`.
        """
        route = data["route"]
        seed = data["seed"]
        det_cfg = self.config.get("detection", {})
        ex_cfg = det_cfg.get("existing_json", {})
        # Resolve base directory
        base_dir = ex_cfg.get("base_dir") or self.config.get("output", {}).get("base_output_dir")
        if not base_dir: 
            print("[ERROR] No base directory resolved for existing detections. Set detection.existing_json.base_dir or output.base_output_dir.")
            return None
        filename = ex_cfg.get("filename", "grounding_detections.json")
        json_path = Path(base_dir) / route / seed / filename
        if not json_path.exists():
            print(f"[WARN] Precomputed detections not found: {json_path}")
            return None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed reading {json_path}: {e}")
            return None
        # Avoid boolean evaluation of numpy arrays (ambiguous truth value)
        frames = data.get("frames")
        if frames is None:
            frames = []
        all_detections: List[Dict[str, Any]] = []
        for fd in payload.get("frame_detections", []):
            idx = int(fd.get("frame_idx", 0))
            # Rebuild image from raw frames
            if 0 <= idx < len(frames):
                frame = frames[idx]
                if isinstance(frame, np.ndarray):
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    image = Image.fromarray(frame)
                else:
                    image = Image.fromarray(np.array(frame))
            else:
                # Out-of-range frame index; skip image but keep detections
                image = None
            all_detections.append({
                "frame_idx": idx,
                "detections": fd.get("detections", []),
                "image": image,
            })
        print(f"Loaded {len(all_detections)} frames of precomputed detections from {json_path}")
        return {
            "route": route,
            "seed": seed,
            "frame_detections": all_detections,
            "frames_processed": len(all_detections),
        }

    # --- Detection via external API servers ---
    def _detect_objects_via_api(self, data: Dict[str, Any], api_url: str) -> Dict[str, Any]:
        frames = data["frames"]
        processing_config = self.config["data"]["processing"]
        detection_config = self.config["detection"]
        all_detections: List[Dict[str, Any]] = []
        max_frames = min(processing_config["max_frames"], len(frames))
        text_prompt = detection_config["text_prompt"]
        timeout = self.config.get("api", {}).get("timeout", 30)
        print(f"[API] Using {api_url} for detection ({max_frames} frames)")
        for frame_idx in range(0, max_frames, processing_config["frame_step"]):
            frame = frames[frame_idx]
            # Ensure uint8 RGB
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            image = Image.fromarray(frame)
            image_b64 = self._frame_to_base64(image)
            try:
                resp = requests.post(
                    f"{api_url}/detect",
                    json={
                        "image": image_b64,
                        "text": text_prompt,
                        "box_threshold": detection_config.get("box_threshold", 0.4),
                        "text_threshold": detection_config.get("text_threshold", 0.3),
                    },
                    timeout=timeout,
                )
                if resp.status_code != 200:
                    err_msg = None
                    try:
                        err_payload = resp.json()
                        err_msg = err_payload.get("error") if isinstance(err_payload, dict) else None
                    except Exception:
                        err_msg = resp.text.strip()[:200]
                    raise RuntimeError(f"HTTP {resp.status_code} {(' - ' + err_msg) if err_msg else ''}")
                result = resp.json()
                detections = []
                for det in result.get("detections", []):
                    box = det.get("box", {})
                    bbox = [box.get("x1", 0.0), box.get("y1", 0.0), box.get("x2", 0.0), box.get("y2", 0.0)]
                    detections.append({
                        "label": det.get("label", "unknown"),
                        "bbox": bbox,
                        "score": float(det.get("confidence", 0.0)),
                    })
                detections = grounding_filter(detections, detection_config.get("box_threshold", 0.4))
            except Exception as e:
                print(f"[API ERROR] {api_url} frame {frame_idx}: {e}")
                detections = []
            all_detections.append({
                "frame_idx": frame_idx,
                "detections": detections,
                "image": image,
            })
            if frame_idx % 50 == 0:
                print(f"[API] Frame {frame_idx}: {len(detections)} objects detected")
        return {
            "route": data["route"],
            "seed": data["seed"],
            "frame_detections": all_detections,
            "frames_processed": len(all_detections),
        }

    # ===== Stage 4: Tracking =====
    def track_objects(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        tracked_results = []
        prev_detections = []
        next_track_id = 0
        tracking_config = self.config["detection"]["tracking"]
        for frame_data in detection_results["frame_detections"]:
            frame_idx = frame_data["frame_idx"]
            detections = frame_data["detections"]
            tracked_detections, next_track_id = simple_object_tracker(
                prev_detections, detections, next_track_id, tracking_config["iou_threshold"]
            )
            tracked_results.append({
                "frame_idx": frame_idx,
                "detections": tracked_detections,
                "image": frame_data["image"],
            })
            prev_detections = tracked_detections
            if frame_idx % 50 == 0:
                print(f"Frame {frame_idx}: {len(tracked_detections)} tracked objects")
        return {
            "route": detection_results["route"],
            "seed": detection_results["seed"],
            "tracked_frames": tracked_results,
            "frames_processed": len(tracked_results),
        }

    # ===== Stage 5: VLM filtering with segment propagation =====
    def vlm_filter_and_propagate(self, tracked_results: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        print("\n=== Stage 5: VLM Filter and Propagate ===")
        actions = data.get("actions")
        stats = data.get("stats")
        all_results: List[Dict[str, Any]] = []
        vlm_responses: List[Dict[str, Any]] = []
        segments: List[Dict[str, Any]] = []
        prev_tracked: List[Dict[str, Any]] = []
        selected_ids: List[int] = []
        segment_start: Optional[int] = None
        frames = tracked_results["tracked_frames"]
        print(f"  Total frames to process: {len(frames)}")
        print(f"  VLM enabled: {self.config['vlm_filtering']['enabled']}")
        # Cache global intent per route
        route = tracked_results.get("route", "")
        gi = self.load_global_info_for_route(route)
        if gi is not None:
            global_intent_override = self.extract_global_intent_from_global_info(gi)
        else:
            global_intent_override = extract_global_intent({}, stats)
        for i, frame_data in enumerate(frames):
            frame_idx = frame_data["frame_idx"]
            tracked_dets = frame_data["detections"]
            image = frame_data["image"]
            if i % 10 == 0:
                print(f"  Processing frame {i}/{len(frames)-1} (frame_idx={frame_idx})...")
            current_action = actions[frame_idx] if actions is not None and frame_idx < len(actions) else None
            action_context = extract_action_intent(
                actions[max(0, frame_idx-5):frame_idx+1] if actions is not None else None
            )
            global_context = global_intent_override
            trigger = is_trigger_by_id(prev_tracked, tracked_dets)
            vlm_result_for_this_trigger = None
            if trigger:
                if segment_start is not None and len(all_results) > 0:
                    segments[-1]["end"] = all_results[-1]["frame_idx"]
                    print(f"    Closed previous segment {len(segments)-1}: frames [{segments[-1]['start']}, {segments[-1]['end']}]")
                segment_start = frame_idx
                print(f"    Starting new segment at frame {segment_start}")
                if self.config["vlm_filtering"]["enabled"]:
                    print(f"    Building VLM prompt with {len(tracked_dets)} detections...")
                    # YAML 可定制的任务上下文
                    task_ctx = (
                        self.config.get("prompt", {}).get("task_context")
                        or ("household manipulation" if getattr(self, "dataset_type", "bench2drive") == "bdv2" else "autonomous driving")
                    )
                    templates = self.config.get("prompt", {}).get("templates", {}) or {}
                    # Use task-level global description as task_context when available (BDV2)
                    task_context_text = None
                    if self.dataset_type == "bdv2":
                        task_context_text = global_context or task_ctx
                    else:
                        task_context_text = task_ctx
                    if self.dataset_type == "bdv2" or (task_ctx and ("manipulation" in task_ctx.lower() or "household" in task_ctx.lower())):
                        vlm_prompt = build_vlm_prompt_topk_bdv2(
                            frame_id=frame_idx,
                            global_context=global_context,
                            action_context=action_context,
                            detections=tracked_dets,
                            template=templates.get("bdv2") or templates.get("manipulation") or templates.get("default"),
                            task_context=task_context_text,
                        )
                    else:
                        vlm_prompt = build_vlm_prompt_topk_bench2drive(
                            frame_id=frame_idx,
                            global_context=global_context,
                            action_context=action_context,
                            detections=tracked_dets,
                            template=templates.get("bench2drive") or templates.get("driving") or templates.get("default"),
                            task_context=task_context_text,
                        )
                    print(f"    VLM prompt preview: {vlm_prompt[:200]}...")
                    try:
                        if image is None:
                            raise RuntimeError("No image available for VLM query on this frame")
                        context = {"text": vlm_prompt, "image": image_to_base64(image)}
                        print("    Calling VLM API...")
                        vlm_text = query_vlm(
                            context,
                            self.config["vlm_filtering"]["detail_level"],
                            model_override=self.config.get("vlm", {}).get("model")
                        )
                        print(f"    VLM response received (length={len(vlm_text) if vlm_text else 0})")
                        vlm_parsed = parse_vlm_topk_response(vlm_text)
                        print(f"    VLM parsed: top_k={len(vlm_parsed.get('top_k', []))}, missing={len(vlm_parsed.get('missing_suspects', []))}")
                        need_re = bool(vlm_parsed.get("missing_suspects"))
                        if need_re and self.config["vlm_filtering"].get("max_redetect_iterations", 0) > 0:
                            suspects = vlm_parsed.get("missing_suspects", [])
                            print(f"    VLM requested redetection for suspects: {suspects}")
                            # Only attempt local redetection if model is available
                            if self.model is not None and self.processor is not None:
                                if self.forward_lock is not None:
                                    with self.forward_lock:
                                        redets = single_frame_redetect(
                                            {"model": self.model, "processor": self.processor},
                                            image,
                                            self.config["detection"]["text_prompt"],
                                            self.device,
                                            self.config["detection"]["box_threshold"],
                                        )
                                else:
                                    redets = single_frame_redetect(
                                        {"model": self.model, "processor": self.processor},
                                        image,
                                        self.config["detection"]["text_prompt"],
                                        self.device,
                                        self.config["detection"]["box_threshold"],
                                    )
                                print(f"    Redetection found {len(redets)} new boxes (not merged; re-querying VLM)")
                                vlm_text = query_vlm(
                                    context,
                                    self.config["vlm_filtering"]["detail_level"],
                                    model_override=self.config.get("vlm", {}).get("model")
                                )
                                vlm_parsed = parse_vlm_topk_response(vlm_text)
                                print(f"    Updated VLM response: top_k={len(vlm_parsed.get('top_k', []))}")
                            else:
                                print("    Skipping redetection: no local model available (API backend or disabled)")
                        vlm_result_for_this_trigger = vlm_parsed
                        vlm_responses.append({"frame_idx": frame_idx, "response": vlm_parsed})
                        selected_ids = [o.get("id") for o in vlm_parsed.get("top_k", []) if "id" in o]
                        print(f"    VLM selected IDs: {selected_ids}")
                    except Exception as e:
                        print(f"    [VLM ERROR] frame {frame_idx}: {e}")
                        selected_ids = []
                        vlm_result_for_this_trigger = {"top_k": [], "missing_suspects": []}
                else:
                    print("    VLM disabled - keeping no objects")
                    selected_ids = []
                segments.append({
                    "start": segment_start,
                    "end": None,
                    "selected_ids": selected_ids.copy(),
                    "vlm": vlm_result_for_this_trigger,
                })
                print(f"    Created segment {len(segments)-1} with {len(selected_ids)} selected IDs")
            filtered = []
            idset = set(selected_ids)
            if idset:
                for det in tracked_dets:
                    tid = det.get("track_id", -1)
                    if tid in idset:
                        filtered.append({
                            "id": tid,
                            "label": det.get("label", "unknown"),
                            "bbox": det.get("bbox", [0, 0, 0, 0]),
                        })
            if i % 10 == 0:
                print(f"    Filtering results: kept {len(filtered)}/{len(tracked_dets)} (selected_ids: {selected_ids})")
            all_results.append({
                "frame_idx": frame_idx,
                "filtered": filtered,
                "action": carla_action_to_text(current_action) if current_action is not None else "",
                "global_context": global_context,
                "action_context": action_context,
            })
            prev_tracked = tracked_dets
        if segments and segments[-1]["end"] is None and all_results:
            segments[-1]["end"] = all_results[-1]["frame_idx"]
            print(f"  Closed final segment {len(segments)-1}: frames [{segments[-1]['start']}, {segments[-1]['end']}]")
        total_kept = sum(len(r["filtered"]) for r in all_results)
        total_detected = sum(len(frames[i]["detections"]) for i in range(len(frames))) if frames else 0
        return {
            "route": tracked_results["route"],
            "seed": tracked_results["seed"],
            "results": all_results,
            "segments": segments,
            "vlm_responses": vlm_responses,
            "frames_processed": len(all_results),
            "summary": {
                "total_kept": total_kept,
                "total_detected": total_detected,
            },
        }

    # ===== Stage 6: Save results =====
    def save_grounding_detections(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Save raw grounding model detections per frame to JSON (no images).

        Always called right after detection, regardless of VLM filtering or tracking.
        """
        save_cfg = self.config["output"]
        base_dir = Path(save_cfg["base_output_dir"]) / detection_results["route"] / detection_results["seed"]
        base_dir.mkdir(parents=True, exist_ok=True)
        fn = save_cfg["filenames"]
        outfile = base_dir / fn.get("grounding_detections", "grounding_detections.json")
        # Strip images and keep only frame_idx + detections
        serializable = {
            "route": detection_results["route"],
            "seed": detection_results["seed"],
            "frames_processed": detection_results.get("frames_processed", 0),
            "frame_detections": [
                {"frame_idx": fd.get("frame_idx"), "detections": fd.get("detections", [])}
                for fd in detection_results.get("frame_detections", [])
            ],
        }
        with open(outfile, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved grounding detections to {outfile}")
        return {"path": str(outfile), "frames": serializable["frames_processed"]}

    # ===== Stage 6: Save results =====
    def save_results(self, vlm_results: Dict[str, Any]) -> Dict[str, Any]:
        save_cfg = self.config["output"]
        base_dir = Path(save_cfg["base_output_dir"]) / vlm_results["route"] / vlm_results["seed"]
        base_dir.mkdir(parents=True, exist_ok=True)
        fn = save_cfg["filenames"]
        result_file = base_dir / fn["vlm_filtered_boxes"]
        with open(result_file, "w") as f:
            json.dump(vlm_results, f, indent=2)
        if vlm_results.get("vlm_responses"):
            vlm_file = base_dir / fn["vlm_responses"]
            with open(vlm_file, "w") as f:
                json.dump(vlm_results["vlm_responses"], f, indent=2)
        frames_processed = vlm_results.get("frames_processed", 0)
        total_kept = sum(len(r.get("filtered", [])) for r in vlm_results.get("results", []))
        summary = {
            "route": vlm_results["route"],
            "seed": vlm_results["seed"],
            "frames_processed": frames_processed,
            "total_kept_boxes": total_kept,
            "avg_kept_per_frame": (total_kept / frames_processed) if frames_processed else 0.0,
            "segments": len(vlm_results.get("segments", [])),
            "vlm_queries": len(vlm_results.get("vlm_responses", [])),
        }
        summary_file = base_dir / fn["vlm_summary"]
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    # ===== Helpers to convert for visualization =====
    @staticmethod
    def convert_tracking_to_viz_format(tracked_results: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        viz_results = []
        actions = data.get("actions", [])
        for frame_data in tracked_results["tracked_frames"]:
            frame_idx = frame_data["frame_idx"]
            tracked_detections = frame_data["detections"]
            filtered_boxes = []
            for det in tracked_detections:
                filtered_boxes.append({
                    "id": det.get("track_id", -1),
                    "label": det.get("label", "unknown"),
                    "bbox": det.get("bbox", [0, 0, 0, 0]),
                })
            action_str = ""
            if actions is not None and frame_idx < len(actions):
                action = actions[frame_idx]
                action_str = carla_action_to_text(action)
            viz_results.append({"frame_idx": frame_idx, "filtered": filtered_boxes, "action": action_str})
        return {
            "route": tracked_results["route"],
            "seed": tracked_results["seed"],
            "results": viz_results,
            "frames_processed": len(viz_results),
        }

    # ===== Stage 7: Visualization (VLM-filtered) =====
    def create_vlm_visualization(self, data: Dict[str, Any], vlm_results: Dict[str, Any]) -> Dict[str, Any]:
        frames = data["frames"]
        route = vlm_results["route"]
        seed = vlm_results["seed"]
        viz_cfg = self.config["visualization"]
        save_cfg = self.config["output"]
        gif_output_path = Path(save_cfg["base_output_dir"]) / save_cfg["directories"]["vlm_gifs"] / route
        gif_output_path.mkdir(parents=True, exist_ok=True)
        annotated_path = Path(save_cfg["base_output_dir"]) / route / seed / save_cfg["directories"]["vlm_annotated_frames"]
        annotated_path.mkdir(parents=True, exist_ok=True)
        box_annotator = sv.BoxAnnotator(color=self.palette, thickness=viz_cfg["annotation"]["box_thickness"])
        label_annotator = sv.LabelAnnotator(
            color=self.palette,
            text_thickness=viz_cfg["annotation"]["label_text_thickness"],
            text_scale=viz_cfg["annotation"]["label_text_scale"],
            text_padding=viz_cfg["annotation"].get("label_text_padding", 1),
        )
        annotated_frames = []
        for result in vlm_results["results"]:
            frame_idx = result["frame_idx"]
            frame = frames[frame_idx].copy()
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            detections_list = result.get("filtered", [])
            if detections_list:
                h, w = frame.shape[:2]
                boxes = []
                labels = []
                ids = []
                for det in detections_list:
                    x1, y1, x2, y2 = det["bbox"]
                    boxes.append([x1 * w, y1 * h, x2 * w, y2 * h])
                    labels.append(det.get("label", ""))
                    ids.append(det.get("id", -1))
                boxes_array = np.array(boxes)
                for idx in range(len(boxes)):
                    box = boxes_array[idx:idx+1]
                    single_det = sv.Detections(xyxy=box, class_id=np.array([0]))
                    frame_bgr = box_annotator.annotate(scene=frame_bgr, detections=single_det)
                    label_text = f"ID{ids[idx]}: {labels[idx]}"
                    frame_bgr = label_annotator.annotate(scene=frame_bgr, detections=single_det, labels=[label_text])
            # Text overlays
            text_cfg = viz_cfg["text_overlay"]
            cv2.putText(
                frame_bgr,
                f"Frame: {frame_idx}",
                (10, frame.shape[0] - 10),
                getattr(cv2, text_cfg["font"]),
                text_cfg["frame_info_scale"],
                tuple(viz_cfg.get("colors", {}).get("text_color", [255, 255, 255])),
                1,
            )
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_file = annotated_path / f"frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(frame_file), frame_bgr)
            annotated_frames.append(frame_rgb)
        gif_file = gif_output_path / save_cfg["gif_templates"]["vlm_filtered"].format(seed=seed)
        if annotated_frames:
            imageio.mimsave(gif_file, annotated_frames, fps=viz_cfg["gif"]["fps"])
            print(f"VLM-filtered GIF saved to {gif_file}")
        return {"gif_path": str(gif_file), "annotated_count": len(annotated_frames), "fps": viz_cfg["gif"]["fps"]}

    # ===== Stage 7b: Visualization (tracking only) =====
    def create_tracking_visualization(self, data: Dict[str, Any], tracking_results: Dict[str, Any]) -> Dict[str, Any]:
        frames = data["frames"]
        route = tracking_results["route"]
        seed = tracking_results["seed"]
        viz_cfg = self.config["visualization"]
        save_cfg = self.config["output"]
        print(f"  Creating tracking-only visualization for {route}/{seed}")
        gif_output_path = Path(save_cfg["base_output_dir"]) / save_cfg["directories"]["tracking_gifs"] / route
        gif_output_path.mkdir(parents=True, exist_ok=True)
        annotated_path = Path(save_cfg["base_output_dir"]) / route / seed / save_cfg["directories"]["tracking_annotated_frames"]
        annotated_path.mkdir(parents=True, exist_ok=True)
        box_annotator = sv.BoxAnnotator(color=self.palette, thickness=viz_cfg["annotation"]["box_thickness"])  # slightly thicker ok
        label_annotator = sv.LabelAnnotator(
            color=self.palette,
            text_thickness=viz_cfg["annotation"]["label_text_thickness"],
            text_scale=viz_cfg["annotation"]["tracking_label_scale"],
            text_padding=viz_cfg["annotation"].get("label_text_padding", 1),
        )
        annotated_frames = []
        for result in tracking_results["results"]:
            frame_idx = result["frame_idx"]
            frame = frames[frame_idx].copy()
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            detections_list = result.get("filtered", [])
            if detections_list:
                h, w = frame.shape[:2]
                boxes = []
                labels = []
                ids = []
                for det in detections_list:
                    x1, y1, x2, y2 = det["bbox"]
                    boxes.append([x1 * w, y1 * h, x2 * w, y2 * h])
                    labels.append(det.get("label", ""))
                    ids.append(det.get("id", -1))
                boxes_array = np.array(boxes)
                for idx in range(len(boxes)):
                    box = boxes_array[idx:idx+1]
                    single_det = sv.Detections(xyxy=box, class_id=np.array([0]))
                    frame_bgr = box_annotator.annotate(scene=frame_bgr, detections=single_det)
                    label_text = f"{ids[idx]}: {labels[idx]}"
                    frame_bgr = label_annotator.annotate(scene=frame_bgr, detections=single_det, labels=[label_text])
            text_cfg = viz_cfg["text_overlay"]
            detection_count = len(detections_list)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_file = annotated_path / f"tracking_frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(frame_file), frame_bgr)
            annotated_frames.append(frame_rgb)
        gif_file = gif_output_path / save_cfg["gif_templates"]["tracking_only"].format(seed=seed)
        if annotated_frames:
            imageio.mimsave(gif_file, annotated_frames, fps=viz_cfg["gif"]["fps"])
            print(f"  Tracking-only GIF saved to {gif_file}")
        return {"gif_path": str(gif_file), "annotated_count": len(annotated_frames), "fps": viz_cfg["gif"]["fps"]}

    # ===== Global info helpers =====
    def load_global_info_for_route(self, route: str) -> Optional[Dict[str, Any]]:
        try:
            global_info_dir = self.config["global_info"]["global_info_dir"]
            if not Path(global_info_dir).is_absolute():
                try:
                    base_dir = Path(__file__).parent
                except NameError:
                    base_dir = Path.cwd()
                global_info_dir = base_dir / global_info_dir
            # BDV2: 任务级全局描述保存在 bdv2/{task}/result.json
            route_key = route
            if self.dataset_type == "bdv2" and isinstance(route, str) and route.startswith("bdv2/"):
                parts = route.split("/")
                if len(parts) >= 2:
                    route_key = "/".join(parts[:2])
            info_path = Path(global_info_dir) / route_key / "result.json"
            if info_path.exists():
                with open(info_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[GLOBAL INFO] Failed to load {route}: {e}")
        return None

    @staticmethod
    def extract_global_intent_from_global_info(info: Dict[str, Any]) -> str:
        desc = info.get("description") or ""
        return desc.strip()

    def build_dynamic_prompt_if_available(self, route: str) -> None:
        gi = self.load_global_info_for_route(route)
        if gi and self.config["global_info"]["use_dynamic_prompts"]:
            # Safely coerce object_types to a Python list of strings (avoid numpy truthiness)
            obj_from_meta = gi.get("metadata", {}).get("object_types", None)
            obj_fallback = gi.get("object_types", None)
            object_types = obj_from_meta if obj_from_meta is not None else obj_fallback
            if object_types is None:
                object_types = []
            elif isinstance(object_types, np.ndarray):
                object_types = [str(x) for x in object_types.tolist()]
            elif not isinstance(object_types, (list, tuple, set)):
                object_types = [str(object_types)]
            phrases = []
            seen = set()
            for obj_type in object_types:
                phrase = obj_type.replace("_", " ").strip()
                if phrase and phrase not in seen:
                    phrases.append(phrase)
                    seen.add(phrase)
            if not phrases:
                phrases = self.config["global_info"]["fallback_objects"]
            prompt = ". ".join(phrases)
            if not prompt.endswith("."):
                prompt += "."
            self.config["detection"]["text_prompt"] = prompt
            print(f"Using dynamic prompt: {prompt}")

    # ===== Top level run for a single (route, seed) =====
    def run_single(self, route: str, seed: str, api_url_override: Optional[str] = None) -> Dict[str, Any]:
        print(f"\n==== Processing {route}/{seed} on {self.device} ====")
        # Only prepare detection backend when we plan to run detection
        use_existing = self.config.get("detection", {}).get("use_existing_json", False)
        if not use_existing:
            # Load model once per thread/pipeline instance (local backend)
            if self.backend == "local":
                if self.model is None or self.processor is None:
                    self.load_model()
            else:
                # Set active API URL for this run (API backend)
                if api_url_override:
                    self._active_api_url = api_url_override
                # If not set, pick a healthy API URL from the list
                if not self._active_api_url:
                    chosen = self._pick_healthy_api_url()
                    if not chosen:
                        raise RuntimeError("No healthy Grounding API server found. Ensure servers are running and api.urls or api.urls_file is configured.")
                    self._active_api_url = chosen
        # Load data
        data = self.load_data(route, seed)
        if data is None:
            return {"route": route, "seed": seed, "success": False, "error": "data load failed"}
        # Dynamic prompt per route
        self.build_dynamic_prompt_if_available(route)
        # Detect or load existing detections
        if use_existing:
            detection_results = self.load_detections_from_json(data)
            if detection_results is None:
                print("[INFO] Falling back to running detection since existing JSON not found.")
                detection_results = self.detect_objects(data, api_url=self._active_api_url)
                self.save_grounding_detections(detection_results)
        else:
            detection_results = self.detect_objects(data, api_url=self._active_api_url)
            # Always save raw grounding detections
            self.save_grounding_detections(detection_results)
        # Track
        tracked_results = self.track_objects(detection_results)
        # VLM or tracking-only
        if self.config["vlm_filtering"]["enabled"]:
            vlm_results = self.vlm_filter_and_propagate(tracked_results, data)
            summary = self.save_results(vlm_results)
            viz = self.create_vlm_visualization(data, vlm_results)
            return {
                "route": route,
                "seed": seed,
                "success": True,
                "frames_processed": summary["frames_processed"],
                "kept_boxes": summary["total_kept_boxes"],
                "gif_created": viz["gif_path"],
            }
        else:
            tracking_viz = self.convert_tracking_to_viz_format(tracked_results, data)
            viz = self.create_tracking_visualization(data, tracking_viz)
            return {
                "route": route,
                "seed": seed,
                "success": True,
                "frames_processed": tracked_results["frames_processed"],
                "tracking_only": True,
                "gif_created": viz["gif_path"],
            }

    def run_single_bdv2(self, task: str, timestamp: str, traj_group: str, traj_name: str, camera: int,
                         api_url_override: Optional[str] = None) -> Dict[str, Any]:
        route = f"bdv2/{task}/{timestamp}/{traj_group}/{traj_name}"
        seed = f"images{camera}"
        print(f"\n==== Processing {route}/{seed} on {self.device} ====")
        use_existing = self.config.get("detection", {}).get("use_existing_json", False)
        if not use_existing:
            if self.backend == "local":
                if self.model is None or self.processor is None:
                    self.load_model()
            else:
                if api_url_override:
                    self._active_api_url = api_url_override
                if not self._active_api_url:
                    chosen = self._pick_healthy_api_url()
                    if not chosen:
                        raise RuntimeError("No healthy Grounding API server found.")
                    self._active_api_url = chosen
        data = self.load_data_bdv2(task, timestamp, traj_group, traj_name, camera)
        if data is None:
            return {"route": route, "seed": seed, "success": False, "error": "data load failed"}
        # Optional dynamic prompt remains (will fallback if no global info)
        self.build_dynamic_prompt_if_available(route)
        if use_existing:
            detection_results = self.load_detections_from_json(data)
            if detection_results is None:
                print("[INFO] Falling back to running detection since existing JSON not found.")
                detection_results = self.detect_objects(data, api_url=self._active_api_url)
                self.save_grounding_detections(detection_results)
        else:
            detection_results = self.detect_objects(data, api_url=self._active_api_url)
            self.save_grounding_detections(detection_results)
        tracked_results = self.track_objects(detection_results)
        if self.config["vlm_filtering"]["enabled"]:
            vlm_results = self.vlm_filter_and_propagate(tracked_results, data)
            summary = self.save_results(vlm_results)
            viz = self.create_vlm_visualization(data, vlm_results)
            return {
                "route": route,
                "seed": seed,
                "success": True,
                "frames_processed": summary["frames_processed"],
                "kept_boxes": summary["total_kept_boxes"],
                "gif_created": viz["gif_path"],
            }
        else:
            tracking_viz = self.convert_tracking_to_viz_format(tracked_results, data)
            viz = self.create_tracking_visualization(data, tracking_viz)
            return {
                "route": route,
                "seed": seed,
                "success": True,
                "frames_processed": tracked_results["frames_processed"],
                "tracking_only": True,
                "gif_created": viz["gif_path"],
            }

    # ===== Internal helpers =====
    def _frame_to_base64(self, image: Image.Image) -> str:
        import io, base64
        if image.mode != "RGB":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _load_api_urls(self) -> List[str]:
        urls = []
        cfg = self.config.get("api", {})
        # Direct list has priority
        for u in cfg.get("urls", []) or []:
            u = str(u).strip()
            if u:
                # Normalize 0.0.0.0 to localhost for client requests
                urls.append(u.replace("://0.0.0.0", "://127.0.0.1"))
        if urls:
            return urls
        # Otherwise try urls_file
        urls_file = cfg.get("urls_file")
        if urls_file:
            try:
                p = Path(urls_file)
                if not p.is_absolute():
                    # default relative to repo root
                    p = (Path(__file__).parent.parent / "grounding_api" / "api_urls.txt").resolve()
                if p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                urls.append(line.replace("://0.0.0.0", "://127.0.0.1"))
            except Exception as e:
                print(f"[API URLS] Failed to read urls_file: {e}")
        return urls

    def _pick_healthy_api_url(self) -> Optional[str]:
        """Probe available API URLs and return the first healthy one."""
        timeout = self.config.get("api", {}).get("timeout", 30)
        for raw_url in self.api_urls:
            url = raw_url.replace("://0.0.0.0", "://127.0.0.1")
            health = None
            try:
                r = requests.get(f"{url}/health", timeout=timeout)
                if r.status_code == 200:
                    try:
                        health = r.json()
                    except Exception:
                        pass
                    if isinstance(health, dict) and health.get("status") == "healthy":
                        print(f"[API] Selected healthy server: {url}")
                        return url
                else:
                    print(f"[API] Health check failed for {url}: HTTP {r.status_code}")
            except Exception as e:
                print(f"[API] Health check error for {url}: {e}")
        return None

    # Auto-launch helpers removed; expect external server management

# Import the stage functions from the original file
# Since we're refactoring, we'll keep the same function signatures but adapt them

def build_grounding_text_prompt_from_global_info(info: Dict[str, Any], config: Dict[str, Any]) -> str:
    obj_from_meta = info.get("metadata", {}).get("object_types", None)
    obj_fallback = info.get("object_types", None)
    object_types = obj_from_meta if obj_from_meta is not None else obj_fallback
    if object_types is None:
        object_types = []
    elif isinstance(object_types, np.ndarray):
        object_types = [str(x) for x in object_types.tolist()]
    elif not isinstance(object_types, (list, tuple, set)):
        object_types = [str(object_types)]
    phrases: List[str] = []
    seen = set()
    for obj_type in object_types:
        phrase = obj_type.replace("_", " ").strip()
        if phrase and phrase not in seen:
            phrases.append(phrase)
            seen.add(phrase)
    if not phrases:
        phrases = config["global_info"]["fallback_objects"]
    prompt = ". ".join(phrases)
    if not prompt.endswith("."):
        prompt += "."
    return prompt

def group_routes(routes_seeds: List[Tuple[int, int]]) -> List[int]:
    """Return unique route IDs in first-seen order."""
    seen = set()
    ordered_routes: List[int] = []
    for r, _ in routes_seeds:
        if r not in seen:
            seen.add(r)
            ordered_routes.append(r)
    return ordered_routes

def seeds_for_route(routes_seeds: List[Tuple[int, int]], route_id: int) -> List[int]:
    return [seed for r, seed in routes_seeds if r == route_id]

def assign_gpus_to_routes(route_ids: List[int]) -> Dict[int, str]:
    """Assign a CUDA device per route. Wrap if fewer GPUs than routes."""
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    mapping: Dict[int, str] = {}
    if num_gpus == 0:
        print("[WARNING] No CUDA devices available. Falling back to CPU for all threads.")
        for r in route_ids:
            mapping[r] = "cpu"
        return mapping
    for idx, r in enumerate(route_ids):
        mapping[r] = f"cuda:{idx % num_gpus}"
    return mapping

# --- Top-level worker for process pool (API backend) ---
def _process_api_worker(config: Dict[str, Any], route: str, seed: str, api_url: str) -> Dict[str, Any]:
    try:
        # Ensure per-process determinism
        processing_config = config["data"]["processing"]
        np.random.seed(processing_config["seed"])
        torch.manual_seed(processing_config["seed"])
        # Force API backend in worker
        cfg = dict(config)
        cfg = json.loads(json.dumps(cfg))  # deep copy via JSON
        cfg.setdefault("detection", {})
        cfg["detection"]["backend"] = "api"
        pipeline = VLMFilterPipeline(cfg)
        res = pipeline.run_single(route, seed, api_url_override=api_url)
        print(f"[Worker] {route}/{seed} via {api_url}: success={res.get('success')}")
        return res
    except Exception as e:
        print(f"[Worker ERROR] {route}/{seed} via {api_url}: {e}")
        return {"route": route, "seed": seed, "success": False, "error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM-filtered detection pipeline (two-domain YAML)")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "vlm_filter_config.yaml"),
                        help="Path to YAML config (supports bench2drive|bdv2 domains)")
    parser.add_argument("--domain", choices=["bench2drive", "bdv2"], default=None,
                        help="Override active domain in config (optional)")
    args = parser.parse_args()

    # Load configuration
    config = load_vlm_filter_config(Path(args.config), domain=args.domain)
    _ = load_pipeline_config()  # ensure pipeline config is cached
    # Set random seeds
    processing_config = config["data"]["processing"]
    np.random.seed(processing_config["seed"])
    torch.manual_seed(processing_config["seed"])
    print("Starting VLM-filtered detection pipeline (YAML configured, OOP & threaded)...")
    print(f"VLM Enabled: {config['vlm_filtering']['enabled']}")
    dataset_type = config.get("dataset", {}).get("type", "bench2drive").lower()
    if dataset_type == "bdv2":
        # BDV2 modes: bdv2_single, bdv2_task, bdv2_all
        mode = config.get("run_mode", {}).get("mode", "bdv2_single")
        if mode == "bdv2_single":
            bd = config.get("bdv2", {})
            single = config.get("run_mode", {}).get("bdv2_single", {})
            task = single.get("task", "open_microwave")
            timestamp = single["timestamp"]
            traj_group = single["traj_group"]
            traj_name = single["traj_name"]
            camera = int(single.get("camera", bd.get("camera", 0)))
            pipeline = VLMFilterPipeline(config, device_override=config["model"].get("device", "cpu"))
            api_url = (pipeline.api_urls[0] if pipeline.backend == "api" and pipeline.api_urls else None)
            result = pipeline.run_single_bdv2(task, timestamp, traj_group, traj_name, camera, api_url_override=api_url)
            print(f"Result: {result}")
        elif mode == "bdv2_task":
            bd = config.get("bdv2", {})
            task = config.get("run_mode", {}).get("bdv2_task", {}).get("task", "open_microwave")
            camera = int(config.get("run_mode", {}).get("bdv2_task", {}).get("camera", bd.get("camera", 0)))
            root = Path(bd.get("dataset_root", "/path/to/dataset/bdv2"))
            # Simple sequential processing over all trajs under task
            pipeline = VLMFilterPipeline(config, device_override=config["model"].get("device", "cpu"))
            api_url = (pipeline.api_urls[0] if pipeline.backend == "api" and pipeline.api_urls else None)
            successes = 0
            total = 0
            for ts_dir in sorted((root / task).iterdir()):
                raw = ts_dir / "raw"
                if not raw.exists():
                    continue
                for g_dir in sorted(d for d in raw.iterdir() if d.is_dir() and d.name.startswith("traj_group")):
                    for tr_dir in sorted(d for d in g_dir.iterdir() if d.is_dir() and d.name.startswith("traj")):
                        total += 1
                        try:
                            res = pipeline.run_single_bdv2(task, ts_dir.name, g_dir.name, tr_dir.name, camera, api_url_override=api_url)
                            if res.get("success"):
                                successes += 1
                        except Exception as e:
                            print(f"[BDV2] ERROR {ts_dir.name}/{g_dir.name}/{tr_dir.name}: {e}")
            print(f"\n==== BDV2 Task Summary ====")
            print(f"Task: {task}, Camera: images{camera}")
            print(f"Total: {total}, Success: {successes}, Failed: {total - successes}")
        elif mode == "bdv2_all":
            bd = config.get("bdv2", {})
            root = Path(bd.get("dataset_root", "/path/to/dataset/bdv2"))
            camera = int(config.get("run_mode", {}).get("bdv2_all", {}).get("camera", bd.get("camera", 0)))
            pipeline = VLMFilterPipeline(config, device_override=config["model"].get("device", "cpu"))
            api_url = (pipeline.api_urls[0] if pipeline.backend == "api" and pipeline.api_urls else None)
            successes = 0
            total = 0
            # Iterate all tasks under dataset_root
            for task_dir in sorted(d for d in root.iterdir() if d.is_dir()):
                task = task_dir.name
                for ts_dir in sorted((root / task).iterdir()):
                    raw = ts_dir / "raw"
                    if not raw.exists():
                        continue
                    for g_dir in sorted(d for d in raw.iterdir() if d.is_dir() and d.name.startswith("traj_group")):
                        for tr_dir in sorted(d for d in g_dir.iterdir() if d.is_dir() and d.name.startswith("traj")):
                            total += 1
                            try:
                                res = pipeline.run_single_bdv2(task, ts_dir.name, g_dir.name, tr_dir.name, camera, api_url_override=api_url)
                                if res.get("success"):
                                    successes += 1
                            except Exception as e:
                                print(f"[BDV2-ALL] ERROR {task}/{ts_dir.name}/{g_dir.name}/{tr_dir.name}: {e}")
            print(f"\n==== BDV2 All Tasks Summary ====")
            print(f"Dataset root: {root}")
            print(f"Total: {total}, Success: {successes}, Failed: {total - successes}")
        else:
            raise SystemExit(f"Invalid run_mode for bdv2: {mode}")
    else:
        run_mode, run_params = get_run_parameters(config)
        routes_seeds = get_routes_seeds()
        if run_mode == "single_seed":
            route = f"route_{run_params['route_id']}"
            seed = f"seed_{run_params['seed_id']}"
            pipeline = VLMFilterPipeline(config, device_override=config["model"].get("device", "cpu"))
            # If API backend, pick the first available URL
            api_url = (pipeline.api_urls[0] if pipeline.backend == "api" and pipeline.api_urls else None)
            result = pipeline.run_single(route, seed, api_url_override=api_url)
            print(f"Result: {result}")
        elif run_mode == "single_route":
            route_id = run_params["route_id"]
            route = f"route_{route_id}"
            # default seeds 200-219
            seed_ids = sorted({seed for r, seed in routes_seeds if r == route_id}) or list(range(200, 220))
            use_existing = config.get("detection", {}).get("use_existing_json", False)
            if use_existing:
                # No detection; just process seeds sequentially for the route
                device = config["model"].get("device", "cpu")
                pipeline = VLMFilterPipeline(config, device_override=device)
                ok = 0
                for idx, seed_id in enumerate(seed_ids, 1):
                    seed = f"seed_{seed_id}"
                    print(f"\n---- Seed {idx}/{len(seed_ids)}: {seed} ----")
                    try:
                        result = pipeline.run_single(route, seed)
                        if result.get("success"):
                            ok += 1
                    except Exception as e:
                        print(f"ERROR processing {route}/{seed}: {e}")
                print(f"\n==== Single Route Summary (precomputed) ====")
                print(f"Route: {route}")
                print(f"Total: {len(seed_ids)}, Success: {ok}, Failed: {len(seed_ids) - ok}")
            elif config.get("detection", {}).get("backend", "local").lower() == "api":
                launcher = VLMFilterPipeline(config)
                api_urls = launcher._load_api_urls()
                if not api_urls:
                    print("[ERROR] API backend selected but no API URLs found. Provide api.urls or a valid api.urls_file.")
                    raise SystemExit(1)
                # Distribute seeds across URLs (round-robin)
                seeds_by_url: Dict[str, List[int]] = {u: [] for u in api_urls}
                for i, sid in enumerate(seed_ids):
                    url = api_urls[i % len(api_urls)]
                    seeds_by_url[url].append(sid)
                # Worker per URL (one thread -> one API)
                results_lock = threading.Lock()
                ok_box = [0]
                def worker(url: str, url_seeds: List[int]):
                    local_ok = 0
                    local_pipe = VLMFilterPipeline(config)
                    # Prevent auto-launch per worker and set URL
                    local_pipe.api_urls = [url]
                    for sid in url_seeds:
                        seed = f"seed_{sid}"
                        try:
                            res = local_pipe.run_single(route, seed, api_url_override=url)
                            if res.get("success"):
                                local_ok += 1
                        except Exception as e:
                            print(f"[Thread {url}] ERROR {route}/{seed}: {e}")
                    with results_lock:
                        ok_box[0] += local_ok
                    print(f"[Thread {url}] Done: {local_ok}/{len(url_seeds)}")
                threads: List[threading.Thread] = []
                for url, url_seeds in seeds_by_url.items():
                    if not url_seeds:
                        continue
                    t = threading.Thread(target=worker, args=(url, url_seeds), daemon=False)
                    t.start()
                    threads.append(t)
                for t in threads:
                    t.join()
                ok = ok_box[0]
                print(f"\n==== Single Route Summary (API threaded) ====")
                print(f"Route: {route}")
                print(f"Total: {len(seed_ids)}, Success: {ok}, Failed: {len(seed_ids) - ok}")
            else:
                device = config["model"].get("device", "cpu")
                pipeline = VLMFilterPipeline(config, device_override=device)
                ok = 0
                for idx, seed_id in enumerate(seed_ids, 1):
                    seed = f"seed_{seed_id}"
                    print(f"\n---- Seed {idx}/{len(seed_ids)}: {seed} ----")
                    try:
                        result = pipeline.run_single(route, seed)
                        if result.get("success"):
                            ok += 1
                    except Exception as e:
                        print(f"ERROR processing {route}/{seed}: {e}")
                print(f"\n==== Single Route Summary ====")
                print(f"Route: {route}")
                print(f"Total: {len(seed_ids)}, Success: {ok}, Failed: {len(seed_ids) - ok}")
        elif run_mode == "all":
            use_existing = config.get("detection", {}).get("use_existing_json", False)
            mt_cfg = config.get("run_mode", {}).get("multithreading", {})
            mt_enabled = mt_cfg.get("enabled", True)
            # Prefer per-route threads; align with "10 routes -> 10 threads"
            if use_existing:
                # Detection is skipped; do not require API servers or local model
                route_ids = group_routes(routes_seeds)
                target_device = config["model"].get("device", "cpu")
                if mt_enabled:
                    print("All routes (tracking+VLM) with precomputed boxes; starting per-route threads...")
                    threads: List[threading.Thread] = []
                    results_lock = threading.Lock()
                    summary_stats: Dict[str, Any] = {"total": 0, "success": 0, "failed": 0}
                    max_threads = int(config.get("run_mode", {}).get("multithreading", {}).get("max_threads", 10))
                    sem = threading.Semaphore(max_threads if max_threads > 0 else len(route_ids))
                    def worker(route_id: int):
                        route = f"route_{route_id}"
                        seed_ids = seeds_for_route(routes_seeds, route_id)
                        local_ok = 0
                        local_pipeline = VLMFilterPipeline(config, device_override=target_device)
                        for seed_id in seed_ids:
                            seed = f"seed_{seed_id}"
                            try:
                                res = local_pipeline.run_single(route, seed)
                                if res.get("success"):
                                    local_ok += 1
                            except Exception as e:
                                print(f"[Thread route_{route_id}] ERROR {seed}: {e}")
                        with results_lock:
                            summary_stats["total"] += len(seed_ids)
                            summary_stats["success"] += local_ok
                            summary_stats["failed"] += (len(seed_ids) - local_ok)
                        print(f"[Thread route_{route_id}] Done: {local_ok}/{len(seed_ids)}")
                    def wrapped_worker(rid: int):
                        try:
                            worker(rid)
                        finally:
                            sem.release()
                    for rid in route_ids:
                        sem.acquire()
                        t = threading.Thread(target=wrapped_worker, args=(rid,), daemon=False)
                        t.start()
                        threads.append(t)
                    for t in threads:
                        t.join()
                    print("\n==== Batch Summary (tracking+VLM, precomputed) ====")
                    print(f"Total: {summary_stats['total']}, Success: {summary_stats['success']}, Failed: {summary_stats['failed']}")
                else:
                    # Sequential per route
                    total = success = 0
                    for route_id in route_ids:
                        route = f"route_{route_id}"
                        seed_ids = seeds_for_route(routes_seeds, route_id)
                        pipeline = VLMFilterPipeline(config, device_override=target_device)
                        for seed_id in seed_ids:
                            total += 1
                            seed = f"seed_{seed_id}"
                            try:
                                res = pipeline.run_single(route, seed)
                                if res.get("success"):
                                    success += 1
                            except Exception as e:
                                print(f"[SEQ route_{route_id}] ERROR {seed}: {e}")
                    print("\n==== Batch Summary (sequential, precomputed) ====")
                    print(f"Total: {total}, Success: {success}, Failed: {total - success}")
            else:
                # Detection required
                if config.get("detection", {}).get("backend", "local").lower() == "api":
                    launcher = VLMFilterPipeline(config)
                    api_urls = launcher._load_api_urls()
                    if not api_urls:
                        print("[ERROR] API backend selected but no API URLs found. Provide api.urls or a valid api.urls_file.")
                        raise SystemExit(1)
                    # Build global task list
                    tasks_all: List[Tuple[str, str]] = []
                    for route_id, seed_id in routes_seeds:
                        tasks_all.append((f"route_{route_id}", f"seed_{seed_id}"))
                    # Distribute tasks across URLs
                    tasks_by_url: Dict[str, List[Tuple[str, str]]] = {u: [] for u in api_urls}
                    for i, task in enumerate(tasks_all):
                        url = api_urls[i % len(api_urls)]
                        tasks_by_url[url].append(task)
                    # Worker per URL
                    results_lock = threading.Lock()
                    total = len(tasks_all)
                    success_box = [0]
                    def worker(url: str, url_tasks: List[Tuple[str, str]]):
                        local_succ = 0
                        local_pipe = VLMFilterPipeline(config)
                        local_pipe.api_urls = [url]
                        for route, seed in url_tasks:
                            try:
                                res = local_pipe.run_single(route, seed, api_url_override=url)
                                if res.get("success"):
                                    local_succ += 1
                            except Exception as e:
                                print(f"[Thread {url}] ERROR {route}/{seed}: {e}")
                        with results_lock:
                            success_box[0] += local_succ
                        print(f"[Thread {url}] Done: {local_succ}/{len(url_tasks)}")
                    threads: List[threading.Thread] = []
                    for url, url_tasks in tasks_by_url.items():
                        if not url_tasks:
                            continue
                        t = threading.Thread(target=worker, args=(url, url_tasks), daemon=False)
                        t.start()
                        threads.append(t)
                    for t in threads:
                        t.join()
                    success = success_box[0]
                    print("\n==== Batch Summary (API threaded) ====")
                    print(f"Total: {total}, Success: {success}, Failed: {total - success}")
                else:
                    # Multi-threaded: each route on its own thread, all share the same GPU and model
                    route_ids = group_routes(routes_seeds)
                    target_device = config["model"].get("device", "cpu")
                    print("All routes share device:")
                    print(f"  device -> {target_device}")
                    shared_lock = threading.Lock()
                    shared_pipeline = VLMFilterPipeline(config, device_override=target_device)
                    shared_pipeline.load_model()
                    shared_model = shared_pipeline.model
                    shared_processor = shared_pipeline.processor
                    threads: List[threading.Thread] = []
                    results_lock = threading.Lock()
                    summary_stats: Dict[str, Any] = {"total": 0, "success": 0, "failed": 0}
                    max_threads = int(config.get("run_mode", {}).get("multithreading", {}).get("max_threads", 10))
                    sem = threading.Semaphore(max_threads if max_threads > 0 else len(route_ids))
                    def worker(route_id: int):
                        route = f"route_{route_id}"
                        seed_ids = seeds_for_route(routes_seeds, route_id)
                        local_ok = 0
                        local_pipeline = VLMFilterPipeline(
                            config,
                            device_override=target_device,
                            shared_processor=shared_processor,
                            shared_model=shared_model,
                            inference_lock=shared_lock,
                        )
                        for seed_id in seed_ids:
                            seed = f"seed_{seed_id}"
                            try:
                                res = local_pipeline.run_single(route, seed)
                                if res.get("success"):
                                    local_ok += 1
                            except Exception as e:
                                print(f"[Thread route_{route_id}] ERROR {seed}: {e}")
                        with results_lock:
                            summary_stats["total"] += len(seed_ids)
                            summary_stats["success"] += local_ok
                            summary_stats["failed"] += (len(seed_ids) - local_ok)
                        print(f"[Thread route_{route_id}] Done: {local_ok}/{len(seed_ids)} succeeded on {target_device}")
                    def wrapped_worker(rid: int):
                        try:
                            worker(rid)
                        finally:
                            sem.release()
                    for rid in route_ids:
                        sem.acquire()
                        t = threading.Thread(target=wrapped_worker, args=(rid,), daemon=False)
                        t.start()
                        threads.append(t)
                    for t in threads:
                        t.join()
                    print("\n==== Batch Summary (threaded) ====")
                    print(f"Total: {summary_stats['total']}, Success: {summary_stats['success']}, Failed: {summary_stats['failed']}")
        else:
            raise ValueError(f"Invalid run mode: {run_mode}")

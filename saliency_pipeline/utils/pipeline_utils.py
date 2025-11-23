#!/usr/bin/env python3
"""
Pipeline Utilities for VLM-Filtered Object Detection

This module provides utility functions and configurations for the autonomous driving
object detection and VLM filtering pipeline. It contains core functions for image
processing, object tracking, VLM communication, and data transformations.

Key Components:
- Configuration loading and management
- Image conversion utilities (PIL/numpy to base64)
- Bounding box operations (IoU calculation, normalization)
- Object detection filtering and tracking
- CARLA action interpretation
- VLM query interface and response parsing
- Global and action intent extraction
- Route and seed configurations (loaded from YAML)

Functions:
- load_pipeline_config(): Load and validate pipeline configuration
- image_to_base64(): Convert images to base64 for VLM input
- calc_IoU(): Calculate intersection over union for bounding boxes
- simple_object_tracker(): IoU-based object tracking across frames
- query_vlm(): Interface to Vision Language Model API
- build_vlm_prompt_topk(): Generate context-aware VLM prompts
- extract_global_intent(): Extract driving intent from route stats
- extract_action_intent(): Extract intent from recent action sequences

Configuration:
- All parameters loaded from YAML configuration files
- Supports environment variable overrides for API keys
"""

from openai import OpenAI
from PIL import Image
import io, base64
import numpy as np
import json
import torch
import os
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

# Global configuration cache
_pipeline_config = None
_client = None

def load_pipeline_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default location.
        
    Returns:
        Dictionary containing pipeline configuration
    """
    global _pipeline_config
    
    if _pipeline_config is not None:
        return _pipeline_config
        
    if config_path is None:
        # Allow override via environment variable for flexibility
        env_path = os.getenv("PIPELINE_CONFIG")
        if env_path:
            config_path = Path(env_path)
        else:
            # Default configuration path relative to this script
            config_path = Path(__file__).parent / "configs" / "pipeline_config.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
        # If YAML uses new domain layout, extract common section
        if isinstance(raw, dict) and any(k in raw for k in ("bench2drive", "bdv2", "common")):
            _pipeline_config = raw.get("common", {}) or {}
        else:
            _pipeline_config = raw
        
    return _pipeline_config

def get_api_client() -> OpenAI:
    """
    Get configured OpenAI client for VLM queries.
    
    Returns:
        Configured OpenAI client instance
    """
    global _client
    
    if _client is not None:
        return _client
        
    config = load_pipeline_config()
    api_config = config["api"]["OpenAI"]
    
    # Get API key from environment variable or use default
    api_key = os.getenv(api_config["api_key_env"], api_config["default_key"])
    
    _client = OpenAI(
        api_key=api_key,
        base_url=api_config["base_url"]
    )
    
    return _client

def get_routes_seeds() -> List[Tuple[int, int]]:
    """
    Get list of (route_id, seed_id) pairs from configuration.
    
    Returns:
        List of tuples containing route and seed IDs
    """
    config = load_pipeline_config()
    return [tuple(pair) for pair in config["routes_seeds"]["pairs"]]

# For backward compatibility
ROUTES_SEEDS = get_routes_seeds()

def image_to_base64(image_input, format_override: Optional[str] = None) -> str:
    """
    Convert image to base64 string, accepts path or PIL Image or numpy array.
    
    Args:
        image_input: Image input (path string, PIL Image, or numpy array)
        format_override: Override image format from config (e.g., 'JPEG', 'WEBP')
        
    Returns:
        Base64 encoded image string
    """
    config = load_pipeline_config()
    img_format = format_override or config["processing"]["image"]["format"]
    
    if isinstance(image_input, str):
        # Path to image
        with Image.open(image_input) as img:
            buf = io.BytesIO()
            img.save(buf, format=img_format)
            return base64.b64encode(buf.getvalue()).decode()
    elif isinstance(image_input, Image.Image):
        # PIL Image
        buf = io.BytesIO()
        image_input.save(buf, format=img_format)
        return base64.b64encode(buf.getvalue()).decode()
    elif isinstance(image_input, np.ndarray):
        # Numpy array
        if image_input.dtype != np.uint8:
            image_input = (image_input * 255).astype(np.uint8)
        img = Image.fromarray(image_input)
        buf = io.BytesIO()
        img.save(buf, format=img_format)
        return base64.b64encode(buf.getvalue()).decode()
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

def calc_IoU(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate IoU between two bounding boxes
    bbox format: [x1, y1, x2, y2] (normalized or pixel coordinates)
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0



def grounding_filter(detections: List[Dict], confidence_threshold: Optional[float] = None) -> List[Dict]:
    """
    Filter detections based on confidence threshold.
    
    Args:
        detections: List of detection dictionaries
        confidence_threshold: Override threshold, uses config default if None
        
    Returns:
        Filtered list of detections
    """
    if confidence_threshold is None:
        config = load_pipeline_config()
        confidence_threshold = config["processing"]["tracking"]["confidence_threshold"]
        
    filtered = []
    for det in detections:
        if "score" in det and det["score"] >= confidence_threshold:
            filtered.append(det)
    return filtered

def normalize_bbox(bbox: List[float], image_shape: Tuple[int, int]) -> List[float]:
    """
    Normalize bbox to [0, 1] range
    bbox: [x1, y1, x2, y2] in pixel coordinates
    image_shape: (height, width)
    """
    h, w = image_shape
    x1, y1, x2, y2 = bbox
    return [x1/w, y1/h, x2/w, y2/h]

def carla_action_to_text(action: np.ndarray) -> str:
    """
    Map CARLA action array to text description using configurable thresholds.
    Action dimensions: [throttle, steer, brake, handbrake, reverse, manual_gear, gear]
    
    Args:
        action: CARLA action array
        
    Returns:
        Text description of vehicle action
    """
    if action is None or len(action) < 3:
        return ""
    
    config = load_pipeline_config()
    thresholds = config["action_processing"]
    
    throttle = action[0] if len(action) > 0 else 0
    steer = action[1] if len(action) > 1 else 0
    brake = action[2] if len(action) > 2 else 0
    handbrake = action[3] if len(action) > 3 else 0
    reverse = action[4] if len(action) > 4 else 0
    
    descriptions = []
    
    # Throttle
    if throttle > thresholds["throttle"]["strong_threshold"]:
        descriptions.append("accelerating strongly")
    elif throttle > thresholds["throttle"]["light_threshold"]:
        descriptions.append("accelerating")
    
    # Steering
    if abs(steer) < thresholds["steering"]["straight_threshold"]:
        descriptions.append("going straight")
    elif steer < -thresholds["steering"]["sharp_turn_threshold"]:
        descriptions.append("turning left sharply")
    elif steer < 0:
        descriptions.append("turning left")
    elif steer > thresholds["steering"]["sharp_turn_threshold"]:
        descriptions.append("turning right sharply")
    else:
        descriptions.append("turning right")
    
    # Braking
    if brake > thresholds["braking"]["hard_threshold"]:
        descriptions.append("braking hard")
    elif brake > thresholds["braking"]["light_threshold"]:
        descriptions.append("braking")
    
    # Special states
    if handbrake > 0.5:
        descriptions.append("handbrake engaged")
    if reverse > 0.5:
        descriptions.append("in reverse")
    
    return ", ".join(descriptions) if descriptions else "Vehicle is idle"


def extract_global_intent(route_info: Optional[Dict] = None, stats: Optional[Dict] = None) -> str:
    """
    Extract global driving intent from route information and statistics.
    
    Args:
        route_info: Route information dictionary (currently unused, for future extension)
        stats: Statistics dictionary containing route completion data
        
    Returns:
        Global driving intent description
    """
    if stats and "scores" in stats:
        completion = stats["scores"].get("score_route", 0)
        if completion < 100:
            return "navigate to destination while avoiding obstacles"
        else:
            return "complete route safely"
    return "navigate safely through urban environment"

def extract_action_intent(actions_window: np.ndarray, window_size: Optional[int] = None) -> str:
    """
    Extract local intent from recent actions using configurable parameters.
    
    Args:
        actions_window: Array of recent actions [window_size, 7]
        window_size: Override window size, uses config default if None
        
    Returns:
        Text description of action intent
    """
    if actions_window is None or len(actions_window) == 0:
        return "maintaining current state"
    
    config = load_pipeline_config()
    if window_size is None:
        window_size = config["action_processing"]["action_window_size"]
    
    # Analyze recent actions
    recent_actions = actions_window[-window_size:] if len(actions_window) > window_size else actions_window
    
    avg_throttle = np.mean(recent_actions[:, 0])
    avg_steer = np.mean(recent_actions[:, 1])
    avg_brake = np.mean(recent_actions[:, 2])
    
    intents = []
    
    # Use configurable thresholds
    thresholds = config["action_processing"]
    
    if avg_brake > 0.3:  # Keep existing logic for intent extraction
        intents.append("preparing to stop")
    elif avg_throttle > thresholds["throttle"]["strong_threshold"]:
        intents.append("accelerating")
    elif avg_throttle < thresholds["throttle"]["light_threshold"]:
        intents.append("maintaining low speed")
    
    if abs(avg_steer) > 0.2:  # Keep existing logic for intent extraction
        if avg_steer < 0:
            intents.append("executing left turn")
        else:
            intents.append("executing right turn")
    
    return " and ".join(intents) if intents else "cruising"

def query_vlm(context: Dict, detail: Optional[str] = None, model_override: Optional[str] = None) -> str:
    """
    Query VLM with an image and text prompt using configured settings.
    
    Args:
        context: Dictionary with 'image' (base64) and 'text' keys
        detail: Override detail level, uses config default if None
        model_override: Override model, uses config default if None
        
    Returns:
        VLM response text
    """
    config = load_pipeline_config()
    client = get_api_client()
    
    if detail is None:
        detail = config["processing"]["vlm"]["detail_level"]
    if model_override is None:
        model_override = config["api"]["models"]["vlm_model"]
    
    base64_str = context["image"]
    img_format = config["processing"]["image"]["format"].lower()
    
    response = client.chat.completions.create(
        model=model_override,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{img_format};base64,{base64_str}",
                            "detail": detail    
                        }
                    },
                    {
                        "type": "text",
                        "text": context["text"]
                    }
                ]
            }
        ]
    )
    
    return response.choices[0].message.content

def simple_object_tracker(prev_detections: List[Dict], curr_detections: List[Dict], 
                         next_track_id: int = 0, iou_threshold: Optional[float] = None) -> Tuple[List[Dict], int]:
    """
    Simple IoU-based object tracker using configurable threshold.
    
    Args:
        prev_detections: Previous frame detections
        curr_detections: Current frame detections
        next_track_id: Next available tracking ID
        iou_threshold: Override IoU threshold, uses config default if None
        
    Returns:
        Tuple of (tracked_detections with track_id, next_track_id)
    """
    if iou_threshold is None:
        config = load_pipeline_config()
        iou_threshold = config["processing"]["tracking"]["iou_threshold"]
        
    tracked = []
    used_curr_indices = set()
    
    # Match current detections with previous ones
    for prev_det in prev_detections:
        best_iou = 0
        best_idx = -1
        
        for idx, curr_det in enumerate(curr_detections):
            if idx in used_curr_indices:
                continue
            
            # Check same category and calculate IoU
            if prev_det["label"] == curr_det["label"]:
                iou = calc_IoU(prev_det["bbox"], curr_det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
        
        # If good match found, keep track_id
        if best_idx >= 0 and best_iou >= iou_threshold:
            tracked_det = curr_detections[best_idx].copy()
            tracked_det["track_id"] = prev_det.get("track_id", next_track_id)
            tracked.append(tracked_det)
            used_curr_indices.add(best_idx)
    
    # Add new detections with new track_ids
    for idx, curr_det in enumerate(curr_detections):
        if idx not in used_curr_indices:
            tracked_det = curr_det.copy()
            tracked_det["track_id"] = next_track_id
            tracked.append(tracked_det)
            next_track_id += 1
    
    return tracked, next_track_id

# ============ New functions for refactored pipeline ============

def is_trigger_by_id(prev_tracked: List[Dict], curr_tracked: List[Dict]) -> bool:
    """
    Trigger if object identity set or class set changed (appear/disappear/class-change).
    Ignore pure motion of same IDs.
    """
    if not prev_tracked:
        return True
    prev_ids = {d.get("track_id") for d in prev_tracked}
    curr_ids = {d.get("track_id") for d in curr_tracked}
    if prev_ids != curr_ids:
        return True
    prev_classes = {d.get("label") for d in prev_tracked}
    curr_classes = {d.get("label") for d in curr_tracked}
    return prev_classes != curr_classes

def _format_candidates(detections: List[Dict]) -> str:
    lines: List[str] = []
    for det in detections:
        tid = det.get("track_id", -1)
        lab = det.get("label", "unknown")
        x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
        lines.append(f"- id:{tid}, class:{lab}, bbox:[{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}]")
    return "\n".join(lines)


def _render_template(template: str,
                     frame_id: int,
                     global_context: str,
                     action_context: str,
                     candidates: str,
                     task_context: Optional[str] = None) -> str:
    """Replace minimal placeholders without interfering with JSON braces.

    Placeholders:
    - {frame_id}
    - {global_context}
    - {action_context}
    - {candidates}
    - {task_context}
    """
    out = (template
           .replace("{frame_id}", str(frame_id))
           .replace("{global_context}", global_context)
           .replace("{action_context}", action_context)
           .replace("{candidates}", candidates))
    return out.replace("{task_context}", task_context or "")

def build_vlm_prompt_topk_bench2drive(frame_id: int,
                                       global_context: str,
                                       action_context: str,
                                       detections: List[Dict],
                                       template: Optional[str] = None,
                                       task_context: Optional[str] = None) -> str:
    """Bench2Drive scenario: Safety/rules/trajectory focus. Supports YAML template override."""
    candidates_text = _format_candidates(detections)
    default_template = (
        "You are analyzing a scene for autonomous driving.\n"
        "Task Context: {task_context}\n\n"
        "Frame: {frame_id}\n"
        "Global Intent: {global_context}\n"
        "Action Intent: {action_context}\n\n"
        "Candidates (normalized xyxy):\n"
        "{candidates}\n\n"
        "Task:\n"
        "1) Select the top-K most relevant objects (<=3 by track_id) to safe driving and current intention.\n"
        "2) For same class but different track_id, pick the one with highest importance now.\n"
        "3) If you suspect missing categories for current frame, list them for re-detection.\n"
        "4) Return JSON with this exact schema:\n\n"
        "{\n"
        "  \"top_k\": [\n"
        "    {\"id\": <int>, \"class\": \"<str>\", \"score\": <float 0~1>, \"rationale\": \"<short>\"}\n"
        "  ],\n"
        "  \"missing_suspects\": [\"<class>\", ...]\n"
        "}\n\n"
        "Rules:\n"
        "- Prioritize collision risks (in-path vehicles, close pedestrians/cyclists), rule-governing items (traffic_light/sign), and trajectory-relevant actors.\n"
        "- 'score' is an importance weight; approximate is fine; we only use ids for filtering.\n"
        "- Return JSON only (no extra text).\n"
    )
    tpl = template or default_template
    return _render_template(tpl, frame_id, global_context, action_context, candidates_text, task_context=task_context)


def build_vlm_prompt_topk_bdv2(frame_id: int,
                               global_context: str,
                               action_context: str,
                               detections: List[Dict],
                               template: Optional[str] = None,
                               task_context: Optional[str] = None) -> str:
    """BDV2 household manipulation scenario: Select key objects/parts around current manipulation action. Supports YAML template override."""
    candidates_text = _format_candidates(detections)
    default_template = (
        "You are analyzing a household manipulation scene.\n"
        "Task Context: {task_context}\n\n"
        "Frame: {frame_id}\n"
        "Global Intent: {global_context}\n"
        "Action Intent: {action_context}\n\n"
        "Candidates (normalized xyxy):\n"
        "{candidates}\n\n"
        "Task:\n"
        "1) Select the top-K most relevant objects (<=3 by track_id) enabling or blocking the current manipulation.\n"
        "2) Prefer direct targets (e.g., door/handle/button/knob), tool/container, immediate support, immediate obstacle.\n"
        "3) If likely target/part is missing, list its class for re-detection.\n"
        "4) Return JSON with this exact schema:\n\n"
        "{\n"
        "  \"top_k\": [\n"
        "    {\"id\": <int>, \"class\": \"<str>\", \"score\": <float 0~1>, \"rationale\": \"<short>\"}\n"
        "  ],\n"
        "  \"missing_suspects\": [\"<class>\", ...]\n"
        "}\n\n"
        "Rules:\n"
        "- Treat parts explicitly when critical (e.g., microwave_handle, cabinet_door, button_panel).\n"
        "- Exclude distant/background items with no role in manipulation.\n"
        "- 'score' is an importance weight; approximate is fine; ids are used for filtering.\n"
        "- Return JSON only (no extra text).\n"
    )
    tpl = template or default_template
    return _render_template(tpl, frame_id, global_context, action_context, candidates_text, task_context=task_context)


def build_vlm_prompt_topk(frame_id: int,
                          global_context: str,
                          action_context: str,
                          detections: List[Dict],
                          task_context: str = "autonomous driving",
                          template: Optional[str] = None) -> str:
    """Compatibility wrapper: Route to corresponding template based on task_context, allows custom template."""
    tc = (task_context or "").lower()
    if "manipulation" in tc or "household" in tc:
        return build_vlm_prompt_topk_bdv2(frame_id, global_context, action_context, detections, template=template, task_context=task_context)
    return build_vlm_prompt_topk_bench2drive(frame_id, global_context, action_context, detections, template=template, task_context=task_context)

def parse_vlm_topk_response(response_text: str) -> Dict:
    """
    Parse VLM response formatted as:
    {"top_k":[{"id":..,"class":"..","score":..,"rationale":".."}, ...],
     "missing_suspects":[...]}
    """
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{.*\}', response_text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {"top_k": [], "missing_suspects": []}

def single_frame_redetect(model_dict: Dict, image_pil, text_prompt: str,
                          device: str, box_threshold: Optional[float] = None) -> List[Dict]:
    """
    Run Grounding DINO on a single PIL image, return filtered detections (normalized bbox).
    
    Args:
        model_dict: Dictionary containing model and processor
        image_pil: PIL Image to process
        text_prompt: Text prompt for detection
        device: Device to run inference on
        box_threshold: Override confidence threshold, uses config default if None
        
    Returns:
        List of detection dictionaries with normalized bboxes
    """
    if box_threshold is None:
        config = load_pipeline_config()
        box_threshold = config["processing"]["tracking"]["confidence_threshold"]
        
    model = model_dict["model"]
    processor = model_dict["processor"]
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt")
    if device.startswith("cuda"):
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image_pil.size[::-1]])
    results = processor.post_process_grounded_object_detection(
        outputs, input_ids=inputs["input_ids"], target_sizes=target_sizes
    )[0]

    detections = []
    if "scores" in results and len(results["scores"]) > 0:
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]
        h, w = image_pil.size[1], image_pil.size[0]
        for box, score, label in zip(boxes, scores, labels):
            if float(score) >= box_threshold:
                detections.append({
                    "label": label,
                    "bbox": [box[0]/w, box[1]/h, box[2]/w, box[3]/h],
                    "score": float(score)
                })
    return detections

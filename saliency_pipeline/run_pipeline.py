#!/usr/bin/env python3
"""
Unified pipeline launcher for Bench2Drive and BDV2.

Reads a single YAML config and orchestrates three stages sequentially:
1) get_global_desc.py
2) vlm_filter.py
3) convert_bbox_to_dataset.py

The script generates per-stage temporary YAML configs from the unified config
and invokes each stage with the correct domain and parameters.

Usage:
  python refactor/run_pipeline.py --config refactor/configs/bdv2/pipeline.yaml
  python refactor/run_pipeline.py --config refactor/configs/bench2drive/pipeline.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(content: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)


def run_cmd(cmd: list[str]) -> int:
    proc = subprocess.Popen(cmd)
    return proc.wait()


def ensure_abs(p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    pp = Path(p)
    return str(pp.resolve()) if not pp.is_absolute() else p


def build_global_desc_config(unified: Dict[str, Any]) -> Dict[str, Any]:
    dataset_type = unified.get("dataset", {}).get("type", "bench2drive").lower()
    out_dir = unified.get("global_desc", {}).get("output_dir", "./global_desc_results")
    out_file = unified.get("global_desc", {}).get("result_filename", "result.json")
    k_frames = int(unified.get("global_desc", {}).get("k_frames", 16))
    api_cfg = unified.get("global_desc", {}).get("api", {})
    # Use OpenAI model for global description
    model_name = unified.get("global_desc", {}).get("model", "Qwen/Qwen2.5-VL-72B-Instruct")
    fallback_models = unified.get("global_desc", {}).get("fallback_models", [])
    api_provider = unified.get("global_desc", {}).get("api_provider", "OpenAI")

    cfg: Dict[str, Any] = {
        "dataset": {"type": dataset_type},
        "data": {
            "processing": {"k_frames": k_frames, "seed": int(unified.get("processing", {}).get("seed", 42))},
        },
        "vlm": {
            "prompt_template": unified.get("global_desc", {}).get("prompt_template", "Describe key objects and context."),
            "model": model_name,
            "fallback_models": fallback_models,
            "api_provider": api_provider,
        },
        "api": {
            "timeout_seconds": int(api_cfg.get("timeout_seconds", 60)),
            "max_retries": int(api_cfg.get("max_retries", 3)),
        },
        # Add shared API configuration
        # Pass only endpoint settings to avoid duplicating model names here
        "shared_api": {"OpenAI": unified.get("api", {}).get("OpenAI", {})},
        "output": {
            "base_output_dir": ensure_abs(out_dir),
            "result_filename": out_file,
            "json_formatting": {
                "indent": int(unified.get("global_desc", {}).get("json_indent", 2)),
                "ensure_ascii": bool(unified.get("global_desc", {}).get("json_ensure_ascii", False)),
            },
        },
        "run_mode": {},
    }

    if dataset_type == "bench2drive":
        b2d = unified.get("dataset", {}).get("bench2drive", {})
        cfg["data"]["data_dir"] = ensure_abs(b2d.get("dataset_dir", ""))
        run_cfg = unified.get("run", {})
        mode = run_cfg.get("mode", "batch")
        cfg["run_mode"]["mode"] = "batch" if mode == "all" else "single_run"
        if mode == "single_seed":
            ss = run_cfg.get("single_seed", {})
            cfg["run_mode"]["single_run"] = {
                "route": f"route_{int(ss['route_id'])}",
                "seed_name": f"seed_{int(ss['seed_id'])}",
            }
    else:
        bdv2 = unified.get("dataset", {}).get("bdv2", {})
        cfg.setdefault("data", {})
        cfg["data"]["bdv2"] = {
            "dataset_root": ensure_abs(bdv2.get("dataset_root", "/path/to/dataset/bdv2")),
            "frame_glob": bdv2.get("frame_glob", "im_*.jpg"),
            "frame_step": int(unified.get("processing", {}).get("frame_step", 1)),
        }
        run_cfg = unified.get("run", {})
        mode = run_cfg.get("mode", "bdv2_single")
        # Map unified bdv2 modes -> stage modes
        if mode == "bdv2_single":
            cfg["run_mode"]["mode"] = "single_traj"
        elif mode == "bdv2_task":
            cfg["run_mode"]["mode"] = "batch_task"
        elif mode == "bdv2_all":
            cfg["run_mode"]["mode"] = "batch_all"
        else:
            cfg["run_mode"]["mode"] = "single_traj"
        if mode == "bdv2_single":
            single = run_cfg.get("bdv2_single", {})
            cfg["run_mode"]["single_traj"] = {
                "task": single["task"],
                "timestamp": single["timestamp"],
                "traj_group": single["traj_group"],
                "traj_name": single["traj_name"],
                "camera": int(single.get("camera", 0)),
            }
        elif mode == "bdv2_task":
            task = run_cfg.get("bdv2_task", {}).get("task")
            camera = int(run_cfg.get("bdv2_task", {}).get("camera", 0))
            cfg["run_mode"]["batch_task"] = {"task": task, "camera": camera}
        elif mode == "bdv2_all":
            camera = int(run_cfg.get("bdv2_all", {}).get("camera", 0))
            cfg["run_mode"]["batch_all"] = {"camera": camera}
        return cfg

    # For bench2drive branch, return cfg after configuration above
    return cfg


def build_vlm_filter_config(unified: Dict[str, Any]) -> Dict[str, Any]:
    dataset_type = unified.get("dataset", {}).get("type", "bench2drive").lower()
    model_id = unified.get("vlm_filter", {}).get("grounding_model_id", "IDEA-Research/grounding-dino-base")
    device = unified.get("vlm_filter", {}).get("device", "cpu")
    det = unified.get("vlm_filter", {}).get("detection", {})
    out = unified.get("vlm_filter", {}).get("output", {})
    api = unified.get("vlm_filter", {}).get("api", {})
    viz = unified.get("vlm_filter", {}).get("visualization", {})
    global_info = unified.get("vlm_filter", {}).get("global_info", {})
    processing = unified.get("processing", {})
    run_cfg = unified.get("run", {})

    # Choose VLM model for filtering: allow override via unified['vlm_filter']['vlm_model'],
    # otherwise reuse global_desc.model to avoid duplicate config knobs.
    vlm_model_for_filter = (
        unified.get("vlm_filter", {}).get("vlm_model")
        or unified.get("global_desc", {}).get("model")
    )

    # task_context selection (YAML-customizable)
    # Priority:
    # 1) unified['vlm_filter']['prompt']['task_context'] as str
    # 2) unified['vlm_filter']['prompt']['task_context'][dataset_type]
    # 3) unified['vlm_filter']['task_context']
    # 4) defaults: bdv2 -> "household manipulation", bench2drive -> "autonomous driving"
    prompt_cfg = unified.get("vlm_filter", {}).get("prompt", {}) or {}
    task_context = None
    if isinstance(prompt_cfg, dict):
        tc = prompt_cfg.get("task_context")
        if isinstance(tc, str) and tc.strip():
            task_context = tc.strip()
        elif isinstance(tc, dict):
            task_context = tc.get(dataset_type) or tc.get("default")
    if not task_context:
        task_context = unified.get("vlm_filter", {}).get("task_context")
    if not task_context:
        task_context = "household manipulation" if dataset_type == "bdv2" else "autonomous driving"

    # Optional prompt templates from unified config
    prompt_cfg = unified.get("vlm_filter", {}).get("prompt", {}) or {}
    templates_cfg = prompt_cfg.get("templates") if isinstance(prompt_cfg, dict) else None

    # YOLO (optional) for gripper detection
    yolo_cfg = (
        unified.get("vlm_filter", {}).get("yolo", {})
        or unified.get("yolo", {})
        or {}
    )

    cfg: Dict[str, Any] = {
        "dataset": {"type": dataset_type},
        "model": {"model_id": model_id, "device": device},
        "vlm": {"model": vlm_model_for_filter} if vlm_model_for_filter else {},
        "prompt": ({"task_context": task_context} | ({"templates": templates_cfg} if templates_cfg else {})),
        "data": {
            "processing": {
                "seed": int(processing.get("seed", 42)),
                "max_frames": int(processing.get("max_frames", 999)),
                "frame_step": int(processing.get("frame_step", 1)),
            }
        },
        "detection": {
            "text_prompt": det.get("text_prompt", "car. person. traffic light. traffic sign. bicycle. motorcycle."),
            "backend": det.get("backend", "local"),
            "use_existing_json": bool(det.get("use_existing_bbox_json", False)),
            "existing_json": {
                "base_dir": ensure_abs(det.get("existing_base_dir", "")),
                "filename": det.get("existing_filename", "grounding_detections.json"),
            },
            "box_threshold": float(det.get("box_threshold", 0.4)),
            "text_threshold": float(det.get("text_threshold", 0.3)),
            "tracking": {"iou_threshold": float(det.get("iou_threshold", 0.2))},
        },
        # Optional YOLO stage for gripper detection
        "yolo": {
            "enabled": bool(yolo_cfg.get("enabled", False)),
            "model_path": ensure_abs(yolo_cfg.get("model_path", str((Path(__file__).parent / "models" / "yolo_widowx_gripper.pt").resolve()))),
            "conf_threshold": float(yolo_cfg.get("conf_threshold", 0.85)),
            # index->class name mapping (default single-class gripper)
            "class_map": yolo_cfg.get("class_map", {0: "gripper"}),
        },
        "vlm_filtering": {
            "enabled": bool(unified.get("vlm_filter", {}).get("vlm_enabled", False)),
            "detail_level": unified.get("vlm_filter", {}).get("detail_level", "low"),
            "max_redetect_iterations": int(unified.get("vlm_filter", {}).get("max_redetect_iterations", 2)),
        },
        "output": {
            "base_output_dir": ensure_abs(out.get("base_output_dir", "./saliency_exp_results")),
            "filenames": {
                "grounding_detections": out.get("grounding_detections", "grounding_detections.json"),
                "vlm_filtered_boxes": out.get("vlm_filtered_boxes", "vlm_filtered_boxes.json"),
                "vlm_responses": out.get("vlm_responses", "vlm_responses.json"),
                "vlm_summary": out.get("vlm_summary", "vlm_summary.json"),
            },
            "directories": {
                "vlm_annotated_frames": viz.get("vlm_annotated", "vlm_annotated"),
                "tracking_annotated_frames": viz.get("tracking_annotated", "tracking_annotated"),
                "vlm_gifs": viz.get("vlm_gifs", "vlm_gifs"),
                "tracking_gifs": viz.get("tracking_gifs", "tracking_gifs"),
            },
            "gif_templates": {
                "vlm_filtered": viz.get("gif_template_vlm", "{seed}_vlm_filtered.gif"),
                "tracking_only": viz.get("gif_template_tracking", "{seed}_tracking_only.gif"),
            },
        },
        "api": {
            "urls": api.get("urls", []),
            "urls_file": ensure_abs(api.get("urls_file")) if api.get("urls_file") else str((Path(__file__).parent.parent / "grounding_api" / "api_urls.txt").resolve()),
            "timeout": int(api.get("timeout", 30)),
            "num_workers": int(api.get("num_workers", 10)),
        },
        "visualization": {
            "gif": {"fps": int(viz.get("fps", 20))},
            "annotation": {
                "box_thickness": int(viz.get("box_thickness", 2)),
                "label_text_thickness": int(viz.get("label_text_thickness", 1)),
                "label_text_scale": float(viz.get("label_text_scale", 0.5)),
                "tracking_label_scale": float(viz.get("tracking_label_scale", 0.5)),
            },
            "text_overlay": {
                "font": viz.get("font", "FONT_HERSHEY_SIMPLEX"),
                "action_text_scale": float(viz.get("action_text_scale", 0.6)),
                "frame_info_scale": float(viz.get("frame_info_scale", 0.5)),
                "tracking_info_scale": float(viz.get("tracking_info_scale", 0.5)),
            },
        },
        # Global info directory resolution:
        # Prefer unified[global_desc.output_dir] so we don't duplicate config.
        # Fallback to explicit vlm_filter.global_info.global_info_dir, then legacy vlm_filter.global_info_dir,
        # finally default to "bdv2_global_info".
        "global_info": {
            "global_info_dir": ensure_abs(
                unified.get("global_desc", {}).get("output_dir")
                or unified.get("vlm_filter", {}).get("global_info", {}).get("global_info_dir")
                or unified.get("vlm_filter", {}).get("global_info_dir")
                or "bdv2_global_info"
            ),
            "use_dynamic_prompts": bool(global_info.get("use_dynamic_prompts", True)),
            "fallback_objects": global_info.get("fallback_objects", [
                "traffic signal", "traffic sign", "vehicle", "pedestrians", "cyclists",
            ]),
        },
        "run_mode": {},
        "bdv2": {},
    }

    if dataset_type == "bench2drive":
        b2d = unified.get("dataset", {}).get("bench2drive", {})
        cfg["data"]["data_dir"] = ensure_abs(b2d.get("dataset_dir", ""))
        mode = run_cfg.get("mode", "all").lower()
        if mode in ("single_seed", "single_route", "all"):
            cfg["run_mode"]["mode"] = mode
            if mode == "single_seed":
                ss = run_cfg.get("single_seed", {})
                cfg["run_mode"]["single_seed"] = {"route_id": int(ss["route_id"]), "seed_id": int(ss["seed_id"])}
            elif mode == "single_route":
                sr = run_cfg.get("single_route", {})
                cfg["run_mode"]["single_route"] = {"route_id": int(sr["route_id"])}
        else:
            cfg["run_mode"]["mode"] = "all"
    else:
        bdv2 = unified.get("dataset", {}).get("bdv2", {})
        cfg["bdv2"] = {
            "dataset_root": ensure_abs(bdv2.get("dataset_root", "/path/to/dataset/bdv2")),
            "frame_glob": bdv2.get("frame_glob", "im_*.jpg"),
            "frame_step": int(processing.get("frame_step", 1)),
        }
        # Add BDV2 data configuration
        cfg["data"]["bdv2"] = {
            "dataset_root": ensure_abs(bdv2.get("dataset_root", "/path/to/dataset/bdv2")),
            "frame_glob": bdv2.get("frame_glob", "im_*.jpg"),
            "frame_step": int(processing.get("frame_step", 1)),
        }
        mode = run_cfg.get("mode", "bdv2_single")
        cfg["run_mode"]["mode"] = mode
        if mode == "bdv2_single":
            single = run_cfg.get("bdv2_single", {})
            cfg["run_mode"]["bdv2_single"] = {
                "task": single["task"],
                "timestamp": single["timestamp"],
                "traj_group": single["traj_group"],
                "traj_name": single["traj_name"],
                "camera": int(single.get("camera", 0)),
            }
        elif mode == "bdv2_task":
            task = run_cfg.get("bdv2_task", {}).get("task")
            camera = int(run_cfg.get("bdv2_task", {}).get("camera", 0))
            cfg["run_mode"]["bdv2_task"] = {"task": task, "camera": camera}
    return cfg


def build_bbox_to_dataset_config(unified: Dict[str, Any]) -> Dict[str, Any]:
    dataset_type = unified.get("dataset", {}).get("type", "bench2drive").lower()
    overwrite = bool(unified.get("bbox_to_dataset", {}).get("overwrite", True))
    saliency_dir = ensure_abs(unified.get("bbox_to_dataset", {}).get("saliency_results_dir") or unified.get("vlm_filter", {}).get("output", {}).get("base_output_dir"))
    filenames = unified.get("bbox_to_dataset", {}).get("filenames", {})

    cfg: Dict[str, Any] = {
        "dataset": {"type": dataset_type},
        "common": {
            "data": {
                "saliency_results_dir": saliency_dir,
                "filenames": {
                    "grounding_detections": filenames.get("grounding_detections", "grounding_detections.json"),
                    "vlm_filtered_boxes": filenames.get("vlm_filtered_boxes", "vlm_filtered_boxes.json"),
                },
                # unified visualization/gif root (optional)
                "visualization": unified.get("bbox_to_dataset", {}).get("visualization", {}),
            },
            "processing": {"overwrite": overwrite},
        },
    }

    run_cfg = unified.get("run", {})
    if dataset_type == "bench2drive":
        b2d = unified.get("dataset", {}).get("bench2drive", {})
        cfg["bench2drive"] = {
            "data": {"dataset_dir": ensure_abs(b2d.get("dataset_dir", ""))},
            "run_mode": {
                "mode": run_cfg.get("mode", "all"),
                "single_seed": run_cfg.get("single_seed", {}),
                "single_route": run_cfg.get("single_route", {}),
            },
        }
    else:
        bdv2 = unified.get("dataset", {}).get("bdv2", {})
        cfg["bdv2"] = {
            "bdv2": {"dataset_root": ensure_abs(bdv2.get("dataset_root", "/path/to/dataset/bdv2"))},
            "run_mode": {"mode": run_cfg.get("mode", "bdv2_single")},
        }
        if run_cfg.get("mode", "bdv2_single") == "bdv2_single":
            cfg["bdv2"]["run_mode"]["bdv2_single"] = dict(run_cfg.get("bdv2_single", {}))
        else:
            cfg["bdv2"]["run_mode"]["bdv2_task"] = dict(run_cfg.get("bdv2_task", {}))
        pkl_names = unified.get("bbox_to_dataset", {}).get("pkl_filenames", {})
        # optional legacy pkls not required anymore
        if pkl_names:
            cfg["common"]["data"]["pkl_filenames"] = pkl_names
    return cfg


def orchestrate(unified_cfg_path: Path, dry_run: bool = False) -> None:
    unified = read_yaml(unified_cfg_path)
    dataset_type = unified.get("dataset", {}).get("type", "bench2drive").lower()
    stages = unified.get("stages", {
        "global_desc": True,
        "vlm_filter": True,
        "bbox_to_dataset": True,
    })

    gen_root = Path(__file__).parent / "configs" / "_generated"
    gen_root.mkdir(parents=True, exist_ok=True)

    # Build configs
    gd_cfg = build_global_desc_config(unified)
    vf_cfg = build_vlm_filter_config(unified)
    bx_cfg = build_bbox_to_dataset_config(unified)

    # Pass unified gif root to converter via common.data.visualization
    try:
        gif_root = unified.get("bbox_to_dataset", {}).get("visualization", {}).get("gif_root", None)
        if gif_root:
            bx_cfg.setdefault("common", {}).setdefault("data", {}).setdefault("visualization", {})["gif_root"] = ensure_abs(gif_root)
    except Exception:
        pass

    # Persist configs (only those needed by enabled stages)
    gd_file = gen_root / f"global_desc__{dataset_type}.yaml"
    vf_file = gen_root / f"vlm_filter__{dataset_type}.yaml"
    bx_file = gen_root / f"bbox_to_dataset__{dataset_type}.yaml"

    print("Generated stage configs:")
    if stages.get("global_desc", True):
        write_yaml(gd_cfg, gd_file)
        print(f"  - {gd_file}")
    if stages.get("vlm_filter", True):
        write_yaml(vf_cfg, vf_file)
        print(f"  - {vf_file}")
    if stages.get("bbox_to_dataset", True):
        write_yaml(bx_cfg, bx_file)
        print(f"  - {bx_file}")

    if dry_run:
        print("Dry-run enabled. Skipping execution.")
        return

    # 1) Global description (optional)
    stage_idx = 1
    if stages.get("global_desc", True):
        gd_cmd = [sys.executable, "-m", "saliency_pipeline.stages.get_global_desc", "--config", str(gd_file), "--domain", dataset_type]
        print(f"\n[{stage_idx}/3] Running global description...")
        rc = run_cmd(gd_cmd)
        if rc != 0:
            raise SystemExit(f"Global description failed with code {rc}")
        stage_idx += 1

    # 2) VLM filter + tracking (optional)
    if stages.get("vlm_filter", True):
        vf_cmd = [sys.executable, "-m", "saliency_pipeline.stages.vlm_filter", "--config", str(vf_file), "--domain", dataset_type]
        print(f"\n[{stage_idx}/3] Running VLM filter pipeline...")
        rc = run_cmd(vf_cmd)
        if rc != 0:
            raise SystemExit(f"VLM filter failed with code {rc}")
        stage_idx += 1

    # 3) Convert bboxes to dataset artifacts (optional)
    if stages.get("bbox_to_dataset", True):
        bx_cmd = [sys.executable, "-m", "saliency_pipeline.stages.convert_bbox_to_dataset", "--config", str(bx_file), "--domain", dataset_type]
        print(f"\n[{stage_idx}/3] Converting bboxes to dataset...")
        rc = run_cmd(bx_cmd)
        if rc != 0:
            raise SystemExit(f"BBox conversion failed with code {rc}")

    print("\nPipeline completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified pipeline launcher for two domains (bench2drive|bdv2)")
    parser.add_argument("--config", required=True, help="Path to unified pipeline YAML")
    parser.add_argument("--dry-run", action="store_true", help="Generate stage configs but do not execute")
    args = parser.parse_args()

    orchestrate(Path(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

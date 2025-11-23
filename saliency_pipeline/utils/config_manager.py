#!/usr/bin/env python3
"""
Config Manager

Provides a unified way to load YAML configs that follow a two-domain layout
(bench2drive and bdv2) plus a common section. It also gracefully handles legacy
configs that don't use the new layout.

Expected YAML layout (new):

active_domain: bench2drive  # optional, can be overridden by function arg
common:
  # keys shared by both domains, e.g. model/output/visualization/api
  ...
bench2drive:
  # domain-specific keys (e.g., data/run_mode/etc.)
  ...
bdv2:
  # domain-specific keys
  ...

load_merged_config() returns a flat dict that merges common + selected domain
and sets dataset.type accordingly so existing code paths keep working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import copy
import yaml


def _deep_update(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update mapping "base" with "overlay" values."""
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _is_domain_layout(data: Dict[str, Any]) -> bool:
    return any(k in data for k in ("bench2drive", "bdv2", "common"))


def load_merged_config(config_path: Path | str, domain: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML config and merge common + selected domain into a flat dict.

    - If config has the new domain layout, returns merged dict and sets
      cfg['dataset']['type'] to the selected domain ('bench2drive' or 'bdv2').
    - If config is legacy style (no domain layout), returns it as-is.

    Args:
        config_path: Path to the YAML file
        domain: Optional domain override. If None, will use 'active_domain' in YAML
                or default to 'bench2drive'.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Legacy config passthrough
    if not isinstance(data, dict) or not _is_domain_layout(data):
        # Ensure legacy configs at least have dataset.type when relevant callers expect it
        if isinstance(data, dict):
            data.setdefault("dataset", {})
            data["dataset"].setdefault("type", data.get("dataset", {}).get("type", "bench2drive"))
        return data

    # Determine selected domain
    selected = (domain or data.get("active_domain") or "bench2drive").lower()
    if selected not in ("bench2drive", "bdv2"):
        raise ValueError(f"Invalid domain: {selected}")

    # Merge common + domain
    merged: Dict[str, Any] = {}
    common = data.get("common") or {}
    dom_cfg = data.get(selected) or {}
    if not isinstance(common, dict):
        raise TypeError("'common' must be a mapping if present")
    if not isinstance(dom_cfg, dict):
        raise TypeError(f"'{selected}' must be a mapping if present")

    _deep_update(merged, common)
    _deep_update(merged, dom_cfg)

    # Ensure dataset.type reflects domain
    merged.setdefault("dataset", {})
    merged["dataset"]["type"] = selected
    return merged


"""Portable configuration and ViT construction helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from aaec.backbones import build_vit_backbone


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def load_config(path: str | Path) -> dict:
    path = Path(path)
    text = os.path.expandvars(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def build_model_from_config(cfg: dict):
    backbone = cfg.get("backbone", {})
    checkpoint_path = backbone.get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path = os.path.expandvars(str(checkpoint_path))
    return build_vit_backbone(
        name=backbone.get("name", "vit_base_patch16_224.augreg_in21k_ft_in1k"),
        num_classes=int(backbone.get("num_classes", 100)),
        pretrained=as_bool(backbone.get("pretrained", True)),
        checkpoint_path=checkpoint_path or None,
    )

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    model = config.get("model", {})
    train = config.get("train", {})
    if not model.get("repo_id") or not model.get("revision"):
        raise ValueError("model.repo_id and model.revision are required")
    if float(model.get("duration_s", 0)) <= 0:
        raise ValueError("model.duration_s must be positive")
    if train:
        if train.get("mode") not in {"baseline", "circular", "plain_adapter"}:
            raise ValueError("train.mode must be baseline, circular, or plain_adapter")
        for key in ("batch_size", "grad_accum", "max_steps", "lora_rank"):
            if int(train.get(key, 0)) <= 0:
                raise ValueError(f"train.{key} must be positive")
        if float(train.get("boundary_weight", 0)) < 0:
            raise ValueError("train.boundary_weight cannot be negative")
    return config

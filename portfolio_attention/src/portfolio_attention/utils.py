"""Utility helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import PathsConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def ensure_output_dirs(paths: PathsConfig) -> None:
    for path in (
        paths.outputs_dir,
        paths.checkpoints_dir,
        paths.metrics_dir,
        paths.logs_dir,
        paths.predictions_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def state_id_from_csv_path(csv_path: Path) -> str:
    name = csv_path.name
    if not name.endswith(".csv"):
        raise ValueError(f"Expected a .csv file, got: {csv_path}")
    stem = csv_path.stem
    suffix = "_panel_long"
    if not stem.endswith(suffix):
        raise ValueError(f"Expected file name ending with '{suffix}.csv', got: {csv_path.name}")
    return stem[: -len(suffix)]


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")

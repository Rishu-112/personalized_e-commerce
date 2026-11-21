"""Shared helpers: config loading, logging, seeding, paths."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def dataset_paths(cfg: dict, dataset: str) -> dict[str, Path]:
    """Resolve raw/processed/artifact directories for 'demo' or 'real'."""
    if dataset not in cfg["datasets"]:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(cfg['datasets'])}")
    d = cfg["datasets"][dataset]
    return {
        "raw": PROJECT_ROOT / d["raw_dir"],
        "processed": PROJECT_ROOT / d["processed_dir"],
        "artifacts": PROJECT_ROOT / d["artifacts_dir"],
        "is_synthetic": d.get("is_synthetic", False),
    }

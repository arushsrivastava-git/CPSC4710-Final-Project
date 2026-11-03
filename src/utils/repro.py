"""Reproducibility helpers: seeding, config loading/hashing, run-environment capture.

Every script in this project goes through these helpers so that each run directory
contains enough information (resolved config, config hash, seeds, library versions,
git commit) to reproduce it and to cite it in the final report.
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + any accelerator)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> dict:
    """Load a YAML experiment config."""
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    """Stable 8-char hash of a config (key order independent).

    Stored with every result row so a number in the report can be traced back to
    the exact configuration that produced it.
    """
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def environment_info() -> dict:
    """Snapshot of the software/hardware environment for the run record."""
    import sklearn

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        # True if tracked files differ from the commit above; results from a dirty
        # tree may not be reproducible from the commit alone.
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def write_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def write_yaml(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

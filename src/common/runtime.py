"""Determinism, thread caps, CPU-only device, and config loading.

Import this before anything that touches numpy/torch RNG. One global SEED; the twin also
seeds its noise per scenario so a scenario is reproducible independently of the run order.
"""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

SEED = 20240101

# Keep BLAS single-threaded: reproducible and kinder to a laptop running the dashboard.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
DATA_CACHE = ROOT / "data_cache"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

for _d in (DATA_CACHE, MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> int:
    """Seed every RNG we use. Returns the seed so callers can log it."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # LSTM kernels have no deterministic path
        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover
        pass
    return seed


def device() -> str:
    """CPU only, by design - the whole project must run on a laptop with no GPU."""
    return "cpu"


@lru_cache(maxsize=None)
def load_config(name: str) -> Dict[str, Any]:
    """Load configs/<name>.yaml (cached). `name` may include or omit the extension."""
    import yaml

    stem = name[:-5] if name.endswith(".yaml") else name
    path = CONFIG_DIR / f"{stem}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_config(name: str, data: Dict[str, Any], header: str = "") -> Path:
    """Write a config back out (used by twin calibration)."""
    import yaml

    stem = name[:-5] if name.endswith(".yaml") else name
    path = CONFIG_DIR / f"{stem}.yaml"
    with path.open("w", encoding="utf-8") as fh:
        if header:
            fh.write(header.rstrip() + "\n\n")
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
    load_config.cache_clear()
    return path

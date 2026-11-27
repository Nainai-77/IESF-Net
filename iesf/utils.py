"""Small shared helpers: seeding, IO, logging, devices."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA) for one estimation stream."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def get_device(name: str = "auto"):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, cls=_NumpyEncoder)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, cls=_NumpyEncoder) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_logger(name: str = "iesf", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

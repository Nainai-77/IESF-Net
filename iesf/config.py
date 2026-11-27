"""Configuration loading: YAML files merged on top of configs/default.yaml."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def deep_update(base: dict, update: dict) -> dict:
    """Recursively merge ``update`` into ``base``.

    A mapping that carries ``__replace__: true`` replaces the target mapping
    instead of being merged into it.
    """
    for key, value in update.items():
        if isinstance(value, dict) and value.get("__replace__"):
            base[key] = {k: copy.deepcopy(v) for k, v in value.items() if k != "__replace__"}
        elif isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _apply_override(cfg: dict, item: str) -> None:
    """Apply a ``dotted.key=value`` override (value parsed as YAML)."""
    key, _, raw = item.partition("=")
    node = cfg
    parts = key.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(path: str | Path | None = None, overrides: Iterable[str] | None = None) -> dict:
    """Load ``default.yaml`` + the model/benchmark/experiment files + an optional override file."""
    cfg = _read_yaml(CONFIG_DIR / "default.yaml")
    cfg["iesf"] = _read_yaml(CONFIG_DIR / "iesf_net.yaml")
    cfg["benchmarks"] = _read_yaml(CONFIG_DIR / "benchmarks.yaml")
    cfg["experiments"] = _read_yaml(CONFIG_DIR / "experiments.yaml")
    if path is not None:
        deep_update(cfg, _read_yaml(Path(path)))
    for item in overrides or []:
        _apply_override(cfg, item)
    cfg["paths"] = {k: str(resolve_path(v)) for k, v in cfg["paths"].items()}
    return cfg


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def market_names(cfg: dict, which: str = "all") -> list[str]:
    if which == "primary":
        return list(cfg["primary_markets"])
    if which == "secondary":
        return list(cfg.get("secondary_markets", []))
    return list(cfg["markets"].keys())


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node

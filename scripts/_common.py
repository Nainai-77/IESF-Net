"""Shared CLI plumbing for the numbered scripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iesf.config import load_config  # noqa: E402


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="YAML merged on top of configs/default.yaml (e.g. configs/smoke.yaml)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides, e.g. optim.max_epochs=50")
    p.add_argument("--markets", nargs="*", default=None, help="restrict to these markets")
    return p


def config_from(args) -> dict:
    return load_config(args.config, args.set)

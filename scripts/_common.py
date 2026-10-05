"""Shared CLI helpers (not part of the public package API)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_seeds(value: str) -> list[int]:
    """Parse one integer or a comma-separated seed list."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]

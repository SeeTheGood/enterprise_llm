"""Ensure project root is on sys.path when running scripts without `pip install -e .`."""
from __future__ import annotations

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)

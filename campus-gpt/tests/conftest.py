"""Shared pytest configuration for the CampusGPT test suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``pytest`` to be run from the repository root or from tests/ directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

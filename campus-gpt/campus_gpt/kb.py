"""Load the verified rules and subject tables from ``kb/``."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

KB_DIR = Path(__file__).resolve().parent.parent / "kb"


def load_rules(path: Path | str | None = None) -> List[Dict[str, Any]]:
    """Return the list of rule records."""
    with open(path or KB_DIR / "rules.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_subjects(path: Path | str | None = None) -> Dict[int, Dict[str, Any]]:
    """Return ``{semester: {"note": str, "courses": [...]}}``."""
    with open(path or KB_DIR / "subjects.yaml", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return {int(k): v for k, v in data["semesters"].items()}

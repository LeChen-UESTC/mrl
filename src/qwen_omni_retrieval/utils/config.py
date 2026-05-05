from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix == ".json":
            return json.load(f)
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "YAML config support requires PyYAML in the existing environment. "
                "Install approval was not requested; use a .json config or add PyYAML manually."
            ) from exc
        data = yaml.safe_load(f)
    return data or {}


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base

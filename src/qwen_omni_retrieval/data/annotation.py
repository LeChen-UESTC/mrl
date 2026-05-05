from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NormalizedRecord:
    sample_id: str
    video_id: str
    vision_caps: list[str]
    subtitle: str | None = None
    vast_cap: str | None = None


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_vision_caps(value: Any) -> list[str]:
    if isinstance(value, list):
        caps = [_clean_text(item) for item in value]
        return [item for item in caps if item]
    cleaned = _clean_text(value)
    return [cleaned] if cleaned else []


def load_annotations(path: str | Path) -> list[NormalizedRecord]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Annotation file must contain a JSON array: {path}")

    records: list[NormalizedRecord] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        video_id = _clean_text(item.get("video_id"))
        vision_caps = normalize_vision_caps(item.get("vision_cap"))
        if not video_id or not vision_caps:
            continue
        records.append(
            NormalizedRecord(
                sample_id=f"{video_id}::{idx}",
                video_id=video_id,
                vision_caps=vision_caps,
                subtitle=_clean_text(item.get("subtitle")),
                vast_cap=_clean_text(item.get("vast_cap")),
            )
        )
    return records

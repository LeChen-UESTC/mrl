from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from qwen_omni_retrieval.data.serialization import decode_jsonable


class CachedRetrievalDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        required_modalities: list[str],
        caption_selection: str = "random",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.required_modalities = required_modalities
        self.caption_selection = caption_selection
        manifest_path = self.cache_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        rows: list[dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                available = set(row.get("available_modalities", []))
                if all(modality in available for modality in required_modalities):
                    rows.append(row)
        if not rows:
            raise ValueError(
                f"No cached rows in {cache_dir} contain all required modalities: {required_modalities}"
            )
        self.rows = rows
        self._loaded_shard_path: Path | None = None
        self._loaded_shard: dict[str, Any] | None = None
        self._loaded_token_shard_path: Path | None = None
        self._loaded_token_shard: dict[str, Any] | None = None
        self._loaded_feature_shard_path: Path | None = None
        self._loaded_feature_shard: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _load_legacy_item(self, row: dict[str, Any]) -> dict[str, Any]:
        shard_path = self.cache_dir / row["cache_shard"]
        if self._loaded_shard_path != shard_path:
            self._loaded_shard = torch.load(shard_path, map_location="cpu")
            self._loaded_shard_path = shard_path
        assert self._loaded_shard is not None
        return self._loaded_shard[row["cache_key"]]

    def _load_token_item(self, row: dict[str, Any]) -> dict[str, Any]:
        shard_path = self.cache_dir / row["token_shard"]
        if self._loaded_token_shard_path != shard_path:
            loaded: dict[str, Any] = {}
            with shard_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    loaded[item["cache_key"]] = decode_jsonable(item["modalities"])
            self._loaded_token_shard = loaded
            self._loaded_token_shard_path = shard_path
        assert self._loaded_token_shard is not None
        return self._loaded_token_shard[row["cache_key"]]

    def _load_feature_item(self, row: dict[str, Any]) -> dict[str, Any]:
        feature_shard = row.get("feature_shard")
        if not feature_shard:
            return {}
        shard_path = self.cache_dir / feature_shard
        if self._loaded_feature_shard_path != shard_path:
            self._loaded_feature_shard = torch.load(shard_path, map_location="cpu")
            self._loaded_feature_shard_path = shard_path
        assert self._loaded_feature_shard is not None
        return self._loaded_feature_shard.get(row["cache_key"], {})

    def _load_item(self, row: dict[str, Any]) -> dict[str, Any]:
        if "cache_shard" in row:
            return self._load_legacy_item(row)
        item = self._load_token_item(row)
        feature_item = self._load_feature_item(row)
        for modality, modality_features in feature_item.items():
            item.setdefault(modality, {}).update(modality_features)
        return item

    def _select_caption(self, candidates: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not candidates:
            raise ValueError("vision_cap cache entry is empty.")
        if self.caption_selection == "first":
            return candidates[0]
        if self.caption_selection == "random":
            return random.choice(candidates)
        raise ValueError(f"Unsupported caption_selection: {self.caption_selection}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        cached = self._load_item(row)
        modalities: dict[str, dict[str, torch.Tensor]] = {}
        for modality in self.required_modalities:
            value = cached[modality]
            if modality == "vision_cap" and isinstance(value, list):
                modalities[modality] = self._select_caption(value)
            else:
                modalities[modality] = value
        return {
            "index": idx,
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "modalities": modalities,
        }

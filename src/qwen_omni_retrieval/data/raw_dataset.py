from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from qwen_omni_retrieval.data.annotation import NormalizedRecord, load_annotations
from qwen_omni_retrieval.data.collator import collate_inputs
from qwen_omni_retrieval.data.media_paths import MediaPaths, resolve_media_paths
from qwen_omni_retrieval.data.preprocess import build_messages, processor_inputs_for_messages


def raw_record_payloads(record: NormalizedRecord, media: MediaPaths) -> dict[str, list[str] | str]:
    payloads: dict[str, list[str] | str] = {
        "vision_cap": record.vision_caps,
        "video": media.video_path,
    }
    if media.audio_path:
        payloads["audio"] = media.audio_path
    if record.subtitle:
        payloads["subtitle"] = record.subtitle
    if record.vast_cap:
        payloads["vast_cap"] = record.vast_cap
    return payloads


class RawRetrievalDataset(Dataset):
    def __init__(
        self,
        *,
        anno_path: str | Path,
        video_dir: str | Path,
        required_modalities: list[str],
        audio_dir: str | Path | None = None,
        use_audio_in_video: bool = False,
        audio_from_video_if_missing: bool = True,
        caption_selection: str = "random",
    ) -> None:
        self.required_modalities = required_modalities
        self.caption_selection = caption_selection
        self.rows: list[dict[str, Any]] = []

        for idx, record in enumerate(load_annotations(anno_path)):
            media = resolve_media_paths(
                record.video_id,
                video_dir,
                audio_dir=audio_dir,
                use_audio_in_video=use_audio_in_video,
                audio_from_video_if_missing=audio_from_video_if_missing,
            )
            payloads = raw_record_payloads(record, media)
            if not all(modality in payloads for modality in required_modalities):
                continue
            self.rows.append(
                {
                    "index": idx,
                    "sample_id": record.sample_id,
                    "video_id": record.video_id,
                    "payloads": payloads,
                    "use_audio_in_video": media.use_audio_in_video,
                }
            )

        if not self.rows:
            raise ValueError(
                f"No raw rows in {anno_path} contain all required modalities: {required_modalities}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def _select_caption(self, candidates: list[str]) -> str:
        if not candidates:
            raise ValueError("vision_cap payload is empty.")
        if self.caption_selection == "first":
            return candidates[0]
        if self.caption_selection == "random":
            return random.choice(candidates)
        raise ValueError(f"Unsupported caption_selection: {self.caption_selection}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        payloads = row["payloads"]
        modalities: dict[str, str] = {}
        for modality in self.required_modalities:
            value = payloads[modality]
            if modality == "vision_cap":
                if not isinstance(value, list):
                    raise TypeError("vision_cap payload must be a list of captions.")
                modalities[modality] = self._select_caption(value)
            else:
                modalities[modality] = str(value)
        return {
            "index": row["index"],
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "use_audio_in_video": row["use_audio_in_video"],
            "modalities": modalities,
        }


class RawRetrievalCollator:
    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "indices": [sample["index"] for sample in samples],
            "sample_ids": [sample["sample_id"] for sample in samples],
            "video_ids": [sample["video_id"] for sample in samples],
            "use_audio_in_video": [sample["use_audio_in_video"] for sample in samples],
            "modalities": [sample["modalities"] for sample in samples],
        }


def raw_batch_to_model_inputs(
    batch: dict[str, Any],
    *,
    required_modalities: list[str],
    processor: Any,
    pad_token_id: int,
    video_nframes: int | None = None,
) -> tuple[dict[str, dict[str, torch.Tensor]], list[int], list[dict[str, str]]]:
    per_modality: dict[str, list[dict[str, torch.Tensor]]] = {
        modality: [] for modality in required_modalities
    }
    valid_positions: list[int] = []
    skipped: list[dict[str, str]] = []

    for position, payloads in enumerate(batch["modalities"]):
        sample_inputs: dict[str, dict[str, torch.Tensor]] = {}
        try:
            for modality in required_modalities:
                use_audio = bool(batch["use_audio_in_video"][position]) if modality == "video" else False
                sample_inputs[modality] = processor_inputs_for_messages(
                    processor,
                    build_messages(
                        modality,
                        payloads[modality],
                        video_nframes=video_nframes if modality == "video" else None,
                    ),
                    use_audio_in_video=use_audio,
                )
        except Exception as exc:
            skipped.append({"video_id": str(batch["video_ids"][position]), "reason": str(exc)})
            continue

        valid_positions.append(position)
        for modality, inputs in sample_inputs.items():
            per_modality[modality].append(inputs)

    if not valid_positions:
        return {}, [], skipped

    collated = {
        modality: collate_inputs(items, pad_token_id=pad_token_id)
        for modality, items in per_modality.items()
    }
    return collated, valid_positions, skipped

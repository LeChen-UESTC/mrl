from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _squeeze_known_batch_dim(key: str, tensor: torch.Tensor) -> torch.Tensor:
    if key in {"input_ids", "attention_mask", "feature_attention_mask"} and tensor.dim() >= 2:
        return tensor.squeeze(0)
    if key == "input_features" and tensor.dim() >= 3 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def _pad_1d(tensors: list[torch.Tensor], pad_value: int | float) -> torch.Tensor:
    max_len = max(t.numel() for t in tensors)
    padded = [F.pad(t, (0, max_len - t.numel()), value=pad_value) for t in tensors]
    return torch.stack(padded, dim=0)


def _pad_last_dim_and_stack(tensors: list[torch.Tensor], pad_value: int | float = 0) -> torch.Tensor:
    max_len = max(t.shape[-1] for t in tensors)
    padded = []
    for tensor in tensors:
        pad_width = max_len - tensor.shape[-1]
        pad = [0, pad_width]
        for _ in range(tensor.dim() - 1):
            pad.extend([0, 0])
        padded.append(F.pad(tensor, tuple(pad), value=pad_value))
    return torch.stack(padded, dim=0)


def _collate_tensor_key(key: str, tensors: list[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    tensors = [_squeeze_known_batch_dim(key, tensor) for tensor in tensors]
    if key == "input_ids":
        return _pad_1d([t.long() for t in tensors], pad_token_id)
    if key in {"attention_mask", "feature_attention_mask"}:
        return _pad_1d([t.long() for t in tensors], 0)
    if key == "input_features":
        return _pad_last_dim_and_stack([t.float() for t in tensors], 0)
    if key in {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}:
        return torch.cat(tensors, dim=0)
    if key == "video_second_per_grid":
        return torch.cat([t.reshape(-1) for t in tensors], dim=0)
    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) == 1:
        return torch.cat(tensors, dim=0) if tensors[0].dim() > 0 and tensors[0].shape[0] == 1 else torch.stack(tensors)
    raise ValueError(f"Do not know how to collate key `{key}` with shapes {sorted(shapes)}")


def collate_inputs(
    inputs: list[dict[str, torch.Tensor]],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    keys = sorted({key for item in inputs for key in item.keys()})
    return {
        key: _collate_tensor_key(key, [item[key] for item in inputs if key in item], pad_token_id)
        for key in keys
        if all(key in item for item in inputs)
    }


class QwenOmniCachedCollator:
    def __init__(self, *, modalities: list[str], pad_token_id: int = 0) -> None:
        self.modalities = modalities
        self.pad_token_id = 0 if pad_token_id is None else pad_token_id

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "indices": [sample["index"] for sample in samples],
            "sample_ids": [sample["sample_id"] for sample in samples],
            "video_ids": [sample["video_id"] for sample in samples],
            "modalities": {},
        }
        for modality in self.modalities:
            batch["modalities"][modality] = collate_inputs(
                [sample["modalities"][modality] for sample in samples],
                pad_token_id=self.pad_token_id,
            )
        return batch

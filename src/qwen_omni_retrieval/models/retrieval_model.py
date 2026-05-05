from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qwen_omni_retrieval.models.embedding import last_token_pool, thinker_final_hidden_state
from qwen_omni_retrieval.models.projection import ProjectionHead


class QwenOmniRetrievalModel(nn.Module):
    def __init__(
        self,
        *,
        thinker: Any,
        projection: ProjectionHead,
        use_audio_in_video_by_modality: dict[str, bool] | None = None,
    ) -> None:
        super().__init__()
        self.thinker = thinker
        self.projection = projection
        self.use_audio_in_video_by_modality = use_audio_in_video_by_modality or {}

    def forward_embedding(self, modality: str, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = thinker_final_hidden_state(
            self.thinker,
            inputs,
            use_audio_in_video=self.use_audio_in_video_by_modality.get(modality, False),
        )
        pooled = last_token_pool(hidden, inputs["attention_mask"].to(hidden.device))
        return self.projection(pooled, modality)

    def forward(
        self,
        modalities: str | dict[str, dict[str, torch.Tensor]],
        inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(modalities, str):
            if inputs is None:
                raise ValueError("inputs must be provided when modalities is a string.")
            return self.forward_embedding(modalities, inputs)
        return {
            modality: self.forward_embedding(modality, modality_inputs)
            for modality, modality_inputs in modalities.items()
        }

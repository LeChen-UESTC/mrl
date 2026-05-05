from __future__ import annotations

import torch

from qwen_omni_retrieval.evaluation.metrics import compute_retrieval_metrics
from qwen_omni_retrieval.losses.gram_volume import gram_volume, volume_to_logits
from qwen_omni_retrieval.losses.contrastive import resolve_loss_mode


def retrieval_score_matrix(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    auxiliary_embeddings: list[torch.Tensor],
    *,
    mode: str | None = None,
    score_mode: str = "inverse_volume",
    scale: float = 10.0,
    temperature: float = 1.0,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, bool]:
    mode = mode or score_mode
    if mode == "cosine":
        return (query_embeddings @ target_embeddings.T) / temperature, True

    volume = gram_volume(query_embeddings, [target_embeddings, *auxiliary_embeddings], eps=eps)
    if mode == "gram":
        return volume, False
    logits = volume_to_logits(
        volume,
        score_mode=mode,
        scale=scale,
        temperature=temperature,
        eps=eps,
    )
    return logits, True


def evaluate_retrieval_from_embeddings(
    *,
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    auxiliary_embeddings: list[torch.Tensor],
    query_ids: list[str],
    target_ids: list[str],
    mode: str | None = None,
    score_mode: str = "inverse_volume",
    scale: float = 10.0,
    temperature: float = 1.0,
    eps: float = 1.0e-6,
) -> dict[str, float]:
    resolved_mode = resolve_loss_mode({"mode": mode or score_mode})
    score_matrix, higher_is_better = retrieval_score_matrix(
        query_embeddings,
        target_embeddings,
        auxiliary_embeddings,
        mode=resolved_mode,
        score_mode=score_mode,
        scale=scale,
        temperature=temperature,
        eps=eps,
    )
    return compute_retrieval_metrics(
        score_matrix,
        query_ids,
        target_ids,
        higher_is_better=higher_is_better,
    )

from __future__ import annotations

import torch
import torch.nn.functional as F

from qwen_omni_retrieval.losses.gram_volume import gram_volume, volume_to_logits
from qwen_omni_retrieval.utils.distributed import all_gather_with_grad, get_rank


LOSS_MODES = {"inverse_volume", "neg_log", "cosine"}


def resolve_loss_mode(loss_cfg: dict | None) -> str:
    loss_cfg = loss_cfg or {}
    mode = loss_cfg.get("mode", loss_cfg.get("score_mode", "inverse_volume"))
    if mode not in LOSS_MODES:
        raise ValueError(f"Unsupported loss mode: {mode}. Supported modes: {sorted(LOSS_MODES)}")
    return mode


def symmetric_gram_contrastive_loss(
    anchor: torch.Tensor,
    candidate_modalities: list[torch.Tensor],
    *,
    score_mode: str = "inverse_volume",
    scale: float = 10.0,
    temperature: float = 1.0,
    eps: float = 1.0e-6,
    label_smoothing: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not candidate_modalities:
        raise ValueError("At least one candidate modality is required.")

    anchor_all = all_gather_with_grad(anchor)
    candidate_all = [all_gather_with_grad(tensor) for tensor in candidate_modalities]

    volume = gram_volume(anchor, candidate_all, eps=eps)
    logits = volume_to_logits(
        volume,
        score_mode=score_mode,
        scale=scale,
        temperature=temperature,
        eps=eps,
    )

    volume_t = gram_volume(anchor_all, candidate_modalities, eps=eps).T
    logits_t = volume_to_logits(
        volume_t,
        score_mode=score_mode,
        scale=scale,
        temperature=temperature,
        eps=eps,
    )

    batch_size = anchor.shape[0]
    labels = torch.arange(batch_size, device=anchor.device, dtype=torch.long)
    labels = labels + get_rank() * batch_size

    loss = (
        F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        + F.cross_entropy(logits_t, labels, label_smoothing=label_smoothing)
    ) / 2.0
    stats = {
        "volume_mean": volume.detach().mean(),
        "logits_mean": logits.detach().mean(),
    }
    return loss, stats


def symmetric_cosine_contrastive_loss(
    anchor: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = 1.0,
    label_smoothing: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    anchor_all = all_gather_with_grad(anchor)
    target_all = all_gather_with_grad(target)

    logits = (anchor @ target_all.T) / temperature
    logits_t = (anchor_all @ target.T).T / temperature

    batch_size = anchor.shape[0]
    labels = torch.arange(batch_size, device=anchor.device, dtype=torch.long)
    labels = labels + get_rank() * batch_size

    loss = (
        F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        + F.cross_entropy(logits_t, labels, label_smoothing=label_smoothing)
    ) / 2.0
    stats = {
        "cosine_logits_mean": logits.detach().mean(),
    }
    return loss, stats


def symmetric_contrastive_loss(
    anchor: torch.Tensor,
    candidate_modalities: list[torch.Tensor],
    *,
    mode: str = "inverse_volume",
    scale: float = 10.0,
    temperature: float = 1.0,
    eps: float = 1.0e-6,
    label_smoothing: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if mode == "cosine":
        if not candidate_modalities:
            raise ValueError("cosine mode requires the primary target modality tensor.")
        return symmetric_cosine_contrastive_loss(
            anchor,
            candidate_modalities[0],
            temperature=temperature,
            label_smoothing=label_smoothing,
        )
    if mode in {"inverse_volume", "neg_log"}:
        return symmetric_gram_contrastive_loss(
            anchor,
            candidate_modalities,
            score_mode=mode,
            scale=scale,
            temperature=temperature,
            eps=eps,
            label_smoothing=label_smoothing,
        )
    raise ValueError(f"Unsupported loss mode: {mode}. Supported modes: {sorted(LOSS_MODES)}")

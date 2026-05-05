from __future__ import annotations

import torch


def gram_volume(
    anchor: torch.Tensor,
    candidate_modalities: list[torch.Tensor],
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Pairwise Gram volume between anchors and aligned candidate modality sets.

    Args:
        anchor: Tensor with shape [n_query, dim].
        candidate_modalities: list of tensors, each with shape [n_candidate, dim].
            Modalities in this list are aligned by candidate index.
        eps: Numerical floor used before sqrt.

    Returns:
        Tensor with shape [n_query, n_candidate]. Smaller values mean the
        anchor and candidate modality vectors span a smaller volume.
    """
    if not candidate_modalities:
        raise ValueError("candidate_modalities must contain at least one tensor.")
    if anchor.dim() != 2:
        raise ValueError(f"anchor must be 2D, got shape {tuple(anchor.shape)}")

    n_query, dim = anchor.shape
    n_candidate = candidate_modalities[0].shape[0]
    for idx, tensor in enumerate(candidate_modalities):
        if tensor.dim() != 2:
            raise ValueError(f"candidate {idx} must be 2D, got {tuple(tensor.shape)}")
        if tensor.shape[1] != dim:
            raise ValueError(f"candidate {idx} dim mismatch: {tensor.shape[1]} != {dim}")
        if tensor.shape[0] != n_candidate:
            raise ValueError("All candidate modality tensors must have the same batch size.")

    vectors = [anchor, *candidate_modalities]
    order = len(vectors)
    dtype = torch.float32
    device = anchor.device
    gram = torch.empty((n_query, n_candidate, order, order), dtype=dtype, device=device)

    anchor_f = anchor.float()
    candidates_f = [tensor.float() for tensor in candidate_modalities]

    anchor_norm = torch.einsum("qd,qd->q", anchor_f, anchor_f).view(n_query, 1)
    gram[:, :, 0, 0] = anchor_norm.expand(n_query, n_candidate)

    for i, cand in enumerate(candidates_f, start=1):
        dots = anchor_f @ cand.T
        gram[:, :, 0, i] = dots
        gram[:, :, i, 0] = dots

    for i, cand_i in enumerate(candidates_f, start=1):
        for j, cand_j in enumerate(candidates_f, start=1):
            dots = torch.einsum("bd,bd->b", cand_i, cand_j).view(1, n_candidate)
            gram[:, :, i, j] = dots.expand(n_query, n_candidate)

    det = torch.linalg.det(gram)
    return torch.sqrt(torch.clamp(det.abs(), min=eps))


def volume_to_logits(
    volume: torch.Tensor,
    *,
    score_mode: str = "inverse_volume",
    scale: float = 10.0,
    temperature: float = 1.0,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    if score_mode == "inverse_volume":
        return scale / (volume + eps)
    if score_mode == "neg_log":
        return -torch.log(volume + eps) / temperature
    raise ValueError(f"Unsupported score_mode: {score_mode}")

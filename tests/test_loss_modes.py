from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_omni_retrieval.evaluation.retrieval import retrieval_score_matrix
from qwen_omni_retrieval.losses.contrastive import (
    resolve_loss_mode,
    resolve_score_mode,
    symmetric_contrastive_loss,
)


def test_resolve_loss_mode_keeps_score_mode_compatibility() -> None:
    assert resolve_loss_mode({"score_mode": "neg_log"}) == "neg_log"
    assert resolve_loss_mode({"mode": "cosine", "score_mode": "inverse_volume"}) == "cosine"


def test_resolve_score_mode_defaults_to_loss_mode() -> None:
    assert resolve_score_mode({"mode": "neg_log"}) == "neg_log"
    assert resolve_score_mode({"mode": "cosine", "score_mode": "inverse_volume"}) == "inverse_volume"


def test_symmetric_cosine_contrastive_loss_is_finite() -> None:
    anchor = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    target = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    loss, stats = symmetric_contrastive_loss(
        anchor,
        [target],
        mode="cosine",
        temperature=1.0,
        label_smoothing=0.0,
    )
    assert torch.isfinite(loss)
    assert "cosine_logits_mean" in stats


def test_cosine_eval_ignores_auxiliary_embeddings() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    aux = [torch.tensor([[0.0, 1.0], [1.0, 0.0]])]
    scores, higher_is_better = retrieval_score_matrix(
        query,
        target,
        aux,
        mode="cosine",
    )
    assert higher_is_better
    assert torch.allclose(scores, torch.eye(2))


if __name__ == "__main__":
    test_resolve_loss_mode_keeps_score_mode_compatibility()
    test_resolve_score_mode_defaults_to_loss_mode()
    test_symmetric_cosine_contrastive_loss_is_finite()
    test_cosine_eval_ignores_auxiliary_embeddings()
    print("ok")

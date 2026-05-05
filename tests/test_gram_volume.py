from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_omni_retrieval.losses.gram_volume import gram_volume, volume_to_logits


def test_gram_volume_shape_and_order() -> None:
    anchor = torch.eye(3)[:2]
    video = torch.eye(3)[:3]
    audio = torch.eye(3)[:3]
    volume = gram_volume(anchor, [video, audio])
    assert volume.shape == (2, 3)
    assert torch.isfinite(volume).all()


def test_inverse_volume_logits() -> None:
    volume = torch.tensor([[1.0, 2.0]])
    logits = volume_to_logits(volume, score_mode="inverse_volume", scale=10.0, eps=0.0)
    assert torch.allclose(logits, torch.tensor([[10.0, 5.0]]))


if __name__ == "__main__":
    test_gram_volume_shape_and_order()
    test_inverse_volume_logits()
    print("ok")

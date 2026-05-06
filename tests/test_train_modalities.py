from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qwen_omni_retrieval.data.modality import normalize_train_modalities, parse_modalities
from train_lora_volume import resolve_training_modalities, training_loss_mode, training_text_anchor


def test_parse_modalities_splits_list_items_with_commas() -> None:
    assert parse_modalities(["video,audio", "vision_cap"]) == ["video", "audio", "vision_cap"]


def test_train_modalities_are_ordered_by_text_anchor_and_video() -> None:
    modalities = normalize_train_modalities(
        ["video", "audio", "vision_cap"],
        dataset_name="vast",
        allow_vast_cap=True,
    )
    assert modalities == ["vision_cap", "video", "audio"]
    assert training_text_anchor(modalities) == "vision_cap"


def test_vast_cap_can_be_the_text_anchor() -> None:
    modalities = normalize_train_modalities(
        ["video", "audio", "vast_cap", "subtitle"],
        dataset_name="vast",
        allow_vast_cap=True,
    )
    assert modalities == ["vast_cap", "video", "audio", "subtitle"]
    assert training_text_anchor(modalities) == "vast_cap"


def test_training_loss_is_chosen_from_modality_count() -> None:
    assert training_loss_mode(["vision_cap", "video"], {"mode": "inverse_volume"}) == "cosine"
    assert training_loss_mode(["vision_cap", "video", "audio"], {"mode": "inverse_volume"}) == "inverse_volume"
    assert training_loss_mode(["vision_cap", "video", "audio", "subtitle"], {"mode": "neg_log"}) == "neg_log"
    assert training_loss_mode(["vision_cap", "video", "audio"], {"mode": "cosine"}) == "inverse_volume"


def test_resolve_training_modalities_prefers_modalities_config() -> None:
    modalities = resolve_training_modalities(
        {"modalities": ["video", "audio", "vision_cap"], "extra_modalities": ["subtitle"]},
        dataset_name="vast",
    )
    assert modalities == ["vision_cap", "video", "audio"]


def test_train_modalities_reject_missing_video_or_multiple_text_anchors() -> None:
    try:
        normalize_train_modalities(["vision_cap", "audio"], dataset_name="vast", allow_vast_cap=True)
    except ValueError as exc:
        assert "`video`" in str(exc)
    else:
        raise AssertionError("missing video should fail")

    try:
        normalize_train_modalities(["vision_cap", "vast_cap", "video"], dataset_name="vast", allow_vast_cap=True)
    except ValueError as exc:
        assert "exactly one text anchor" in str(exc)
    else:
        raise AssertionError("multiple text anchors should fail")


if __name__ == "__main__":
    test_parse_modalities_splits_list_items_with_commas()
    test_train_modalities_are_ordered_by_text_anchor_and_video()
    test_vast_cap_can_be_the_text_anchor()
    test_training_loss_is_chosen_from_modality_count()
    test_resolve_training_modalities_prefers_modalities_config()
    test_train_modalities_reject_missing_video_or_multiple_text_anchors()
    print("ok")

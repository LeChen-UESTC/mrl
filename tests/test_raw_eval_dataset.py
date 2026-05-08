from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import qwen_omni_retrieval.data.raw_dataset as raw_dataset_module
from qwen_omni_retrieval.data.raw_dataset import (
    RawRetrievalCollator,
    RawRetrievalDataset,
    raw_batch_to_model_inputs,
)
from train_lora_volume import eval_video_fps, eval_video_nframes, str_to_bool


def test_raw_retrieval_dataset_resolves_dotted_video_and_audio_ids() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        anno_path = root / "anno.json"
        video_dir = root / "videos"
        audio_dir = root / "audios"
        video_dir.mkdir()
        audio_dir.mkdir()
        (video_dir / "TYBUpSwGryk.28.mp4").touch()
        (audio_dir / "TYBUpSwGryk.28.mp3").touch()
        anno_path.write_text(
            json.dumps(
                [
                    {
                        "video_id": "TYBUpSwGryk.28",
                        "vision_cap": ["first caption", "second caption"],
                        "subtitle": "subtitle text",
                    }
                ]
            ),
            encoding="utf-8",
        )

        dataset = RawRetrievalDataset(
            anno_path=anno_path,
            video_dir=video_dir,
            audio_dir=audio_dir,
            required_modalities=["vision_cap", "video", "audio", "subtitle"],
            use_audio_in_video=False,
            audio_from_video_if_missing=False,
            caption_selection="first",
        )

        sample = dataset[0]
        assert sample["video_id"] == "TYBUpSwGryk.28"
        assert sample["modalities"]["vision_cap"] == "first caption"
        assert sample["modalities"]["video"] == str(video_dir / "TYBUpSwGryk.28.mp4")
        assert sample["modalities"]["audio"] == str(audio_dir / "TYBUpSwGryk.28.mp3")
        assert sample["modalities"]["subtitle"] == "subtitle text"


def test_raw_batch_to_model_inputs_collates_processor_outputs() -> None:
    original_processor_inputs = raw_dataset_module.processor_inputs_for_messages
    seen_video_items: list[dict[str, Any]] = []

    def fake_processor_inputs(
        processor: Any,
        messages: list[dict[str, Any]],
        *,
        use_audio_in_video: bool,
    ) -> dict[str, torch.Tensor]:
        del processor
        content = messages[1]["content"]
        if content[0]["type"] == "video":
            seen_video_items.append(content[0])
        token_count = 3 + int(use_audio_in_video) + int(content[0]["type"] == "video")
        return {
            "input_ids": torch.arange(token_count).reshape(1, token_count),
            "attention_mask": torch.ones(1, token_count, dtype=torch.long),
        }

    raw_dataset_module.processor_inputs_for_messages = fake_processor_inputs
    try:
        batch = RawRetrievalCollator()(
            [
                {
                    "index": 0,
                    "sample_id": "sample0",
                    "video_id": "video0",
                    "use_audio_in_video": True,
                    "modalities": {
                        "vision_cap": "caption",
                        "video": "/tmp/video0.mp4",
                    },
                }
            ]
        )
        inputs, valid_positions, skipped = raw_batch_to_model_inputs(
            batch,
            required_modalities=["vision_cap", "video"],
            processor=object(),
            pad_token_id=0,
            video_fps=2.0,
        )
    finally:
        raw_dataset_module.processor_inputs_for_messages = original_processor_inputs

    assert valid_positions == [0]
    assert skipped == []
    assert set(inputs) == {"vision_cap", "video"}
    assert inputs["vision_cap"]["input_ids"].shape == (1, 3)
    assert inputs["video"]["input_ids"].shape == (1, 5)
    assert seen_video_items == [{"type": "video", "video": "/tmp/video0.mp4", "fps": 2.0}]


def test_training_eval_helpers_parse_bool_and_nframes() -> None:
    assert str_to_bool("false", default=True) is False
    assert str_to_bool("true", default=False) is True
    assert eval_video_nframes({"nframes": 2}) == 2
    assert eval_video_nframes({"video": {"nframes": "4"}}) == 4
    assert eval_video_nframes({"nframes": ""}) is None
    assert eval_video_fps({"fps": 2}) == 2.0


if __name__ == "__main__":
    test_raw_retrieval_dataset_resolves_dotted_video_and_audio_ids()
    test_raw_batch_to_model_inputs_collates_processor_outputs()
    test_training_eval_helpers_parse_bool_and_nframes()
    print("ok")

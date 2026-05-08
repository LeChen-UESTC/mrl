from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_omni_retrieval.data.preprocess import (
    build_messages,
    normalize_fps,
    normalize_log_every,
    normalize_nframes,
    normalize_optional_positive_int,
    output_dir_with_frame_suffix,
    output_dir_with_sampling_suffix,
)


def test_nframes_none_means_no_limit() -> None:
    assert normalize_nframes(None) is None
    assert normalize_nframes("") is None
    messages = build_messages("video", "/tmp/video.mp4")
    video_item = messages[1]["content"][0]
    assert "nframes" not in video_item


def test_video_message_can_carry_nframes() -> None:
    messages = build_messages("video", "/tmp/video.mp4", video_nframes=8)
    video_item = messages[1]["content"][0]
    assert video_item["nframes"] == 8


def test_video_message_can_carry_fps() -> None:
    assert normalize_fps("2") == 2.0
    messages = build_messages("video", "/tmp/video.mp4", video_fps=2.0)
    video_item = messages[1]["content"][0]
    assert video_item["fps"] == 2.0


def test_video_sampling_rejects_nframes_and_fps_together() -> None:
    try:
        build_messages("video", "/tmp/video.mp4", video_nframes=8, video_fps=2.0)
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("nframes and fps together should fail")


def test_output_dir_gets_frame_suffix_once() -> None:
    assert output_dir_with_frame_suffix("/tmp/vast_train", 8) == Path("/tmp/vast_train_n_frames_8")
    assert output_dir_with_frame_suffix("/tmp/vast_train_n_frames_8", 8) == Path(
        "/tmp/vast_train_n_frames_8"
    )
    assert output_dir_with_frame_suffix("/tmp/vast_train", None) == Path("/tmp/vast_train")


def test_output_dir_gets_fps_suffix_once() -> None:
    assert output_dir_with_sampling_suffix("/tmp/msrvtt_test", fps=2.0) == Path("/tmp/msrvtt_test_fps_2")
    assert output_dir_with_sampling_suffix("/tmp/msrvtt_test_fps_2", fps=2.0) == Path(
        "/tmp/msrvtt_test_fps_2"
    )


def test_optional_positive_int_parsing() -> None:
    assert normalize_optional_positive_int(None, name="max_samples") is None
    assert normalize_optional_positive_int("", name="max_samples") is None
    assert normalize_optional_positive_int("10", name="max_samples") == 10


def test_log_every_parsing() -> None:
    assert normalize_log_every(None) == 100
    assert normalize_log_every("") == 100
    assert normalize_log_every("0") == 0
    assert normalize_log_every("25") == 25


if __name__ == "__main__":
    test_nframes_none_means_no_limit()
    test_video_message_can_carry_nframes()
    test_video_message_can_carry_fps()
    test_video_sampling_rejects_nframes_and_fps_together()
    test_output_dir_gets_frame_suffix_once()
    test_output_dir_gets_fps_suffix_once()
    test_optional_positive_int_parsing()
    test_log_every_parsing()
    print("ok")

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_retrieval import parse_args
from eval_retrieval import resolve_eval_cache_dir
from eval_retrieval import resolve_projection_config
from qwen_omni_retrieval.utils.naming import (
    checkpoint_model_and_step,
    dataset_dir_name,
    default_eval_output_json,
    frame_sample_suffix,
    sampling_cache_dir,
)


def test_checkpoint_model_and_step_uses_parent_model_dir() -> None:
    assert checkpoint_model_and_step(
        "/mnt/d/cl/mrl/outputs/models/train_vast_inverse_volume_video-audio-vision_cap_lr5e-5_lora-r16-a32-d0.05_proj-shared-1024/step_0001000"
    ) == (
        "train_vast_inverse_volume_video-audio-vision_cap_lr5e-5_lora-r16-a32-d0.05_proj-shared-1024",
        "step_0001000",
    )


def test_dataset_dir_name_normalizes_msrvtt() -> None:
    assert dataset_dir_name("msrvtt") == "msr_vtt"
    assert dataset_dir_name("msr_vtt") == "msr_vtt"


def test_default_eval_output_json_uses_zero_shot_dataset_dir() -> None:
    output_path = default_eval_output_json(
        {
            "name": "msr_vtt",
            "checkpoint_dir": "/mnt/d/cl/mrl/outputs/models/train_vast_inverse_volume_video-audio-vision_cap_lr5e-5_lora-r16-a32-d0.05_proj-shared-1024/step_0001000",
            "nframes": 8,
        }
    )
    assert output_path == (
        "/mnt/d/cl/mrl/outputs/eval/zero_shot/msr_vtt/"
        "train_vast_inverse_volume_video-audio-vision_cap_lr5e-5_lora-r16-a32-d0.05_proj-shared-1024_step_0001000_8frames.json"
    )


def test_frame_sample_suffix_defaults_empty_nframes_to_2fps() -> None:
    assert frame_sample_suffix({"nframes": None}) == "2fps"
    assert frame_sample_suffix({"nframes": ""}) == "2fps"
    assert default_eval_output_json(
        {
            "name": "didemo",
            "checkpoint_dir": "/mnt/d/cl/mrl/outputs/models/model_a/step_0000001",
            "nframes": "",
        }
    ) == "/mnt/d/cl/mrl/outputs/eval/zero_shot/didemo/model_a_step_0000001_2fps.json"


def test_frame_sample_suffix_supports_fps() -> None:
    assert frame_sample_suffix({"fps": 2}) == "2fps"
    assert default_eval_output_json(
        {
            "name": "msr_vtt",
            "checkpoint_dir": "/mnt/d/cl/mrl/outputs/models/model_a/step_0000001",
            "fps": 1.5,
        }
    ) == "/mnt/d/cl/mrl/outputs/eval/zero_shot/msr_vtt/model_a_step_0000001_1.5fps.json"


def test_sampling_cache_dir_uses_requested_suffix(tmp_path: Path) -> None:
    base = tmp_path / "msrvtt_test"
    assert sampling_cache_dir(base, {"nframes": 12}) == tmp_path / "msrvtt_test_n_frames_12"
    assert sampling_cache_dir(base, {"fps": 2}) == tmp_path / "msrvtt_test_fps_2"


def test_resolve_eval_cache_dir_falls_back_to_raw_when_requested_cache_missing(tmp_path: Path) -> None:
    base = tmp_path / "msrvtt_test"
    cache_dir, reason = resolve_eval_cache_dir({"cache_dir": str(base), "nframes": 12})
    assert cache_dir == tmp_path / "msrvtt_test_n_frames_12"
    assert reason is not None
    assert "using raw data" in reason


def test_resolve_eval_cache_dir_uses_requested_cache_when_present(tmp_path: Path) -> None:
    base = tmp_path / "msrvtt_test"
    requested = tmp_path / "msrvtt_test_fps_2"
    requested.mkdir()
    cache_dir, reason = resolve_eval_cache_dir({"cache_dir": str(base), "fps": 2})
    assert cache_dir == requested
    assert reason is None


def test_resolve_projection_config_prefers_checkpoint_train_config(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "model" / "step_0001500"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "train_config.json").write_text(
        json.dumps({"projection": {"mode": "none", "embed_dim": 1024, "normalize": True}}),
        encoding="utf-8",
    )

    projection_cfg, source, config_path = resolve_projection_config(
        {
            "projection": {"mode": "shared", "embed_dim": 1024, "normalize": True},
            "eval": {"checkpoint_dir": str(checkpoint_dir)},
        },
        projection_state={"head.weight": torch.empty(1024, 3584)},
    )

    assert projection_cfg["mode"] == "none"
    assert projection_cfg["embed_dim"] == 1024
    assert source == "checkpoint_train_config"
    assert config_path == str(checkpoint_dir / "train_config.json")


def test_resolve_projection_config_can_infer_none_from_empty_state_dict(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "model" / "step_0001500"
    checkpoint_dir.mkdir(parents=True)

    projection_cfg, source, config_path = resolve_projection_config(
        {
            "projection": {"mode": "shared", "embed_dim": 1024, "normalize": False},
            "eval": {"checkpoint_dir": str(checkpoint_dir)},
        },
        projection_state={},
    )

    assert projection_cfg == {"mode": "none", "normalize": False}
    assert source == "projection_state_dict"
    assert config_path is None


def test_resolve_projection_config_falls_back_to_eval_config(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "model" / "step_0001500"
    checkpoint_dir.mkdir(parents=True)

    projection_cfg, source, config_path = resolve_projection_config(
        {
            "projection": {"mode": "shared", "embed_dim": 1024, "normalize": True},
            "eval": {"checkpoint_dir": str(checkpoint_dir)},
        },
        projection_state=None,
    )

    assert projection_cfg == {"mode": "shared", "embed_dim": 1024, "normalize": True}
    assert source == "eval_config"
    assert config_path is None


def test_parse_args_requires_batch_size() -> None:
    old_argv = sys.argv
    stderr = io.StringIO()
    try:
        sys.argv = ["eval_retrieval.py", "--config", "configs/eval/msrvtt.yaml"]
        try:
            with contextlib.redirect_stderr(stderr):
                parse_args()
        except SystemExit as exc:
            assert exc.code == 2
            assert "--batch_size" in stderr.getvalue()
        else:
            raise AssertionError("missing --batch_size should fail")
    finally:
        sys.argv = old_argv


def test_eval_sampling_cli_args_are_mutually_exclusive() -> None:
    old_argv = sys.argv
    stderr = io.StringIO()
    try:
        sys.argv = [
            "eval_retrieval.py",
            "--config",
            "configs/eval/msrvtt.yaml",
            "--batch_size",
            "4",
            "--nframes",
            "12",
            "--fps",
            "2",
        ]
        try:
            with contextlib.redirect_stderr(stderr):
                parse_args()
        except SystemExit as exc:
            assert exc.code == 2
            assert "not allowed with argument" in stderr.getvalue()
        else:
            raise AssertionError("--nframes and --fps together should fail")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    test_checkpoint_model_and_step_uses_parent_model_dir()
    test_dataset_dir_name_normalizes_msrvtt()
    test_default_eval_output_json_uses_zero_shot_dataset_dir()
    test_frame_sample_suffix_defaults_empty_nframes_to_2fps()
    test_frame_sample_suffix_supports_fps()
    with contextlib.ExitStack() as stack:
        import tempfile

        tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        test_sampling_cache_dir_uses_requested_suffix(tmp_path)
        test_resolve_eval_cache_dir_falls_back_to_raw_when_requested_cache_missing(tmp_path)
        test_resolve_eval_cache_dir_uses_requested_cache_when_present(tmp_path)
        test_resolve_projection_config_prefers_checkpoint_train_config(tmp_path / "proj_config")
        test_resolve_projection_config_can_infer_none_from_empty_state_dict(tmp_path / "proj_state")
        test_resolve_projection_config_falls_back_to_eval_config(tmp_path / "proj_eval")
    test_parse_args_requires_batch_size()
    test_eval_sampling_cli_args_are_mutually_exclusive()
    print("ok")

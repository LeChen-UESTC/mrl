from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_omni_retrieval.utils.naming import (
    checkpoint_model_and_step,
    dataset_dir_name,
    default_eval_output_json,
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
        }
    )
    assert output_path == (
        "/mnt/d/cl/mrl/outputs/eval/zero_shot/msr_vtt/"
        "train_vast_inverse_volume_video-audio-vision_cap_lr5e-5_lora-r16-a32-d0.05_proj-shared-1024_step_0001000.json"
    )


if __name__ == "__main__":
    test_checkpoint_model_and_step_uses_parent_model_dir()
    test_dataset_dir_name_normalizes_msrvtt()
    test_default_eval_output_json_uses_zero_shot_dataset_dir()
    print("ok")

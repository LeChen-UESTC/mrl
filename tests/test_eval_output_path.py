from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_retrieval import checkpoint_output_suffix, output_json_with_checkpoint_suffix


def test_checkpoint_output_suffix_uses_last_two_path_parts() -> None:
    assert (
        checkpoint_output_suffix("/mnt/d/cl/mrl/outputs/vast_lora_volume/step_0001000")
        == "vast_lora_volume_step_0001000"
    )


def test_output_json_gets_checkpoint_suffix() -> None:
    assert output_json_with_checkpoint_suffix(
        "/mnt/d/cl/mrl/outputs/eval_msrvtt.json",
        "/mnt/d/cl/mrl/outputs/vast_lora_volume/step_0000000",
    ) == "/mnt/d/cl/mrl/outputs/eval_msrvtt_vast_lora_volume_step_0000000.json"


def test_output_json_suffix_is_not_duplicated() -> None:
    output_path = "/mnt/d/cl/mrl/outputs/eval_msrvtt_vast_lora_volume_step_0000000.json"
    assert output_json_with_checkpoint_suffix(
        output_path,
        "/mnt/d/cl/mrl/outputs/vast_lora_volume/step_0000000",
    ) == output_path


if __name__ == "__main__":
    test_checkpoint_output_suffix_uses_last_two_path_parts()
    test_output_json_gets_checkpoint_suffix()
    test_output_json_suffix_is_not_duplicated()
    print("ok")

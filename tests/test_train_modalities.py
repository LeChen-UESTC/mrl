from __future__ import annotations

from argparse import Namespace
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qwen_omni_retrieval.data.modality import normalize_train_modalities, parse_modalities
from train_lora_volume import (
    apply_cli_overrides,
    prepare_training_names,
    resolve_training_modalities,
    total_train_steps,
    training_loss_mode,
    training_text_anchor,
)


def make_args(**overrides: object) -> Namespace:
    defaults = {
        "modality": None,
        "extra_modalities": None,
        "epochs": None,
        "max_steps": None,
        "batch_size": None,
        "eval_batch_size": None,
        "eval_nframes": None,
        "num_workers": None,
        "learning_rate": None,
        "weight_decay": None,
        "max_grad_norm": None,
        "log_steps": None,
        "save_steps": None,
        "eval_steps": None,
        "do_eval": None,
        "loss_mode": None,
        "wandb_mode": None,
        "lora_r": None,
        "lora_alpha": None,
        "lora_dropout": None,
        "lora_target_modules": None,
        "lora_bias": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


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


def test_apply_cli_overrides_training_and_lora_args() -> None:
    cfg = {
        "training": {"modalities": ["vision_cap", "video"], "extra_modalities": ["audio"]},
        "eval_datasets": [{"name": "msr_vtt", "nframes": ""}, {"name": "didemo"}],
        "loss": {"mode": "inverse_volume"},
        "wandb": {},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
    }
    apply_cli_overrides(
        cfg,
        make_args(
            modality=["video", "audio", "vision_cap"],
            epochs=3,
            max_steps=10000,
            batch_size=2,
            eval_batch_size=4,
            eval_nframes=8,
            num_workers=6,
            learning_rate=5.0e-5,
            weight_decay=0.02,
            max_grad_norm=0.5,
            log_steps=5,
            save_steps=250,
            eval_steps=250,
            do_eval="false",
            loss_mode="neg_log",
            wandb_mode="offline",
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            lora_target_modules=["q_proj", "v_proj"],
            lora_bias="none",
        ),
    )

    assert cfg["training"]["modalities"] == ["video", "audio", "vision_cap"]
    assert "extra_modalities" not in cfg["training"]
    assert cfg["training"]["epochs"] == 3
    assert cfg["training"]["max_steps"] == 10000
    assert cfg["training"]["batch_size"] == 2
    assert cfg["training"]["eval_batch_size"] == 4
    assert [item["nframes"] for item in cfg["eval_datasets"]] == [8, 8]
    assert cfg["training"]["num_workers"] == 6
    assert cfg["training"]["learning_rate"] == 5.0e-5
    assert cfg["training"]["weight_decay"] == 0.02
    assert cfg["training"]["max_grad_norm"] == 0.5
    assert cfg["training"]["log_steps"] == 5
    assert cfg["training"]["save_steps"] == 250
    assert cfg["training"]["eval_steps"] == 250
    assert cfg["training"]["do_eval"] is False
    assert cfg["loss"]["mode"] == "neg_log"
    assert "score_mode" not in cfg["loss"]
    assert cfg["wandb"]["mode"] == "offline"
    assert cfg["lora"]["r"] == 32
    assert cfg["lora"]["alpha"] == 64
    assert cfg["lora"]["dropout"] == 0.1
    assert cfg["lora"]["target_modules"] == ["q_proj", "v_proj"]
    assert cfg["lora"]["bias"] == "none"


def test_total_train_steps_respects_epoch_count_and_step_cap() -> None:
    assert total_train_steps(epochs=3, batches_per_epoch=100, max_steps=0) == 300
    assert total_train_steps(epochs=3, batches_per_epoch=100, max_steps=120) == 120
    assert total_train_steps(epochs=1, batches_per_epoch=100, max_steps=200) == 100


def test_prepare_training_names_sets_model_output_dir_and_wandb_name() -> None:
    cfg = {
        "training": {
            "dataset_name": "vast",
            "modalities": ["video", "audio", "vision_cap"],
            "learning_rate": 5.0e-5,
        },
        "loss": {"mode": "inverse_volume"},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
        "projection": {"mode": "shared", "embed_dim": 1024},
        "wandb": {"name": "old-name"},
    }
    run_name = prepare_training_names(
        cfg,
        dataset_name="vast",
        train_modalities=["vision_cap", "video", "audio"],
    )
    expected = "train_vast_inverse_volume_video-audio-vision_cap_lr5e-5_lora-r16-a32-d0.05_proj-shared-1024"
    assert run_name == expected
    assert cfg["training"]["model_name"] == expected
    assert cfg["training"]["output_dir"] == f"/mnt/d/cl/mrl/outputs/models/{expected}"
    assert cfg["wandb"]["name"] == expected


if __name__ == "__main__":
    test_parse_modalities_splits_list_items_with_commas()
    test_train_modalities_are_ordered_by_text_anchor_and_video()
    test_vast_cap_can_be_the_text_anchor()
    test_training_loss_is_chosen_from_modality_count()
    test_resolve_training_modalities_prefers_modalities_config()
    test_train_modalities_reject_missing_video_or_multiple_text_anchors()
    test_apply_cli_overrides_training_and_lora_args()
    test_total_train_steps_respects_epoch_count_and_step_cap()
    test_prepare_training_names_sets_model_output_dir_and_wandb_name()
    print("ok")

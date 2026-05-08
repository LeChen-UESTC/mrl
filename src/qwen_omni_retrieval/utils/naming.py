from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_MODEL_OUTPUT_ROOT = Path("/mnt/d/cl/mrl/outputs/models")
DEFAULT_ZERO_SHOT_EVAL_ROOT = Path("/mnt/d/cl/mrl/outputs/eval/zero_shot")

DATASET_DIR_NAMES = {
    "msrvtt": "msr_vtt",
    "msr-vtt": "msr_vtt",
    "msr_vtt": "msr_vtt",
}


def sanitize_name(value: Any) -> str:
    text = str(value)
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)
    return cleaned.strip("_") or "none"


def compact_float(value: Any) -> str:
    if value is None or value == "":
        return "none"
    number = float(value)
    text = f"{number:.8g}"
    if "e" in text or "E" in text:
        mantissa, exponent = text.lower().split("e", 1)
        sign = ""
        if exponent.startswith(("+", "-")):
            sign = exponent[0]
            exponent = exponent[1:]
        exponent = exponent.lstrip("0") or "0"
        text = f"{mantissa}e{sign}{exponent}"
    return text


def compact_lr(value: Any) -> str:
    if value is None or value == "":
        return "none"
    number = float(value)
    if number != 0 and abs(number) < 1.0e-3:
        mantissa, exponent = f"{number:.8e}".split("e", 1)
        mantissa = mantissa.rstrip("0").rstrip(".")
        sign = ""
        if exponent.startswith(("+", "-")):
            sign = exponent[0]
            exponent = exponent[1:]
        exponent = exponent.lstrip("0") or "0"
        return f"{mantissa}e{sign}{exponent}"
    return compact_float(number)


def dataset_dir_name(dataset_name: str) -> str:
    normalized = str(dataset_name).strip().lower()
    return DATASET_DIR_NAMES.get(normalized, sanitize_name(dataset_name))


def modality_segment(modalities: list[str]) -> str:
    return "-".join(sanitize_name(modality) for modality in modalities) or "none"


def lora_segment(lora_cfg: dict[str, Any]) -> str:
    return (
        f"lora-r{sanitize_name(lora_cfg.get('r', 'none'))}"
        f"-a{sanitize_name(lora_cfg.get('alpha', 'none'))}"
        f"-d{sanitize_name(compact_float(lora_cfg.get('dropout')))}"
    )


def projection_segment(projection_cfg: dict[str, Any]) -> str:
    mode = sanitize_name(projection_cfg.get("mode", "shared"))
    embed_dim = projection_cfg.get("embed_dim")
    if embed_dim is None or embed_dim == "":
        return f"proj-{mode}"
    return f"proj-{mode}-{sanitize_name(embed_dim)}"


def training_run_name(
    *,
    dataset_name: str,
    loss_mode: str,
    modalities: list[str],
    learning_rate: Any,
    lora_cfg: dict[str, Any],
    projection_cfg: dict[str, Any],
) -> str:
    parts = [
        f"train_{sanitize_name(dataset_name)}",
        sanitize_name(loss_mode),
        modality_segment(modalities),
        f"lr{sanitize_name(compact_lr(learning_rate))}",
        lora_segment(lora_cfg),
        projection_segment(projection_cfg),
    ]
    return "_".join(parts)


def checkpoint_model_and_step(checkpoint_dir: str | Path | None) -> tuple[str, str]:
    if not checkpoint_dir:
        return "zero_shot", "step_0000000"
    checkpoint_path = Path(str(checkpoint_dir).rstrip("/"))
    step_name = sanitize_name(checkpoint_path.name)
    if checkpoint_path.name.startswith("step_") and checkpoint_path.parent.name:
        return sanitize_name(checkpoint_path.parent.name), step_name
    return sanitize_name(checkpoint_path.name), "step_unknown"


def frame_sample_suffix(eval_cfg: dict[str, Any]) -> str:
    nframes = eval_cfg.get("nframes")
    if nframes is None and isinstance(eval_cfg.get("video"), dict):
        nframes = eval_cfg["video"].get("nframes")
    if nframes is None or nframes == "":
        return "2fps"
    return f"{sanitize_name(nframes)}frames"


def default_eval_output_json(eval_cfg: dict[str, Any]) -> str:
    dataset_name = eval_cfg.get("name", "eval")
    output_root = Path(eval_cfg.get("output_root", DEFAULT_ZERO_SHOT_EVAL_ROOT))
    model_name, step_name = checkpoint_model_and_step(eval_cfg.get("checkpoint_dir"))
    frame_suffix = frame_sample_suffix(eval_cfg)
    return str(output_root / dataset_dir_name(dataset_name) / f"{model_name}_{step_name}_{frame_suffix}.json")

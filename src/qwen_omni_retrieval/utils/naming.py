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


def normalize_sampling_nframes(value: Any) -> int | None:
    if value is None or value == "":
        return None
    number = int(value)
    if number <= 0:
        raise ValueError(f"nframes must be a positive integer when set, got {number}.")
    return number


def normalize_sampling_fps(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    if number <= 0:
        raise ValueError(f"fps must be positive when set, got {number}.")
    return number


def sampling_values(config: dict[str, Any]) -> tuple[int | None, float | None]:
    nframes = config.get("nframes")
    fps = config.get("fps")
    video_cfg = config.get("video")
    if isinstance(video_cfg, dict):
        if nframes is None:
            nframes = video_cfg.get("nframes")
        if fps is None:
            fps = video_cfg.get("fps")
    normalized_nframes = normalize_sampling_nframes(nframes)
    normalized_fps = normalize_sampling_fps(fps)
    if normalized_nframes is not None and normalized_fps is not None:
        raise ValueError("nframes and fps are mutually exclusive; specify only one.")
    return normalized_nframes, normalized_fps


def sampling_cache_suffix(config: dict[str, Any]) -> str:
    nframes, fps = sampling_values(config)
    if nframes is not None:
        return f"_n_frames_{nframes}"
    if fps is not None:
        return f"_fps_{compact_float(fps)}"
    return ""


def sampling_description(config: dict[str, Any]) -> str:
    nframes, fps = sampling_values(config)
    if nframes is not None:
        return f"nframes={nframes}"
    if fps is not None:
        return f"fps={compact_float(fps)}"
    return "processor default fps"


def sampling_cache_dir(cache_dir: str | Path, config: dict[str, Any]) -> Path:
    cache_path = Path(cache_dir)
    suffix = sampling_cache_suffix(config)
    if not suffix or cache_path.name.endswith(suffix):
        return cache_path
    return cache_path.with_name(f"{cache_path.name}{suffix}")


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
    nframes, fps = sampling_values(eval_cfg)
    if nframes is not None:
        return f"{sanitize_name(nframes)}frames"
    if fps is not None:
        return f"{sanitize_name(compact_float(fps))}fps"
    return "2fps"


def default_eval_output_json(eval_cfg: dict[str, Any]) -> str:
    dataset_name = eval_cfg.get("name", "eval")
    output_root = Path(eval_cfg.get("output_root", DEFAULT_ZERO_SHOT_EVAL_ROOT))
    model_name, step_name = checkpoint_model_and_step(eval_cfg.get("checkpoint_dir"))
    frame_suffix = frame_sample_suffix(eval_cfg)
    return str(output_root / dataset_dir_name(dataset_name) / f"{model_name}_{step_name}_{frame_suffix}.json")

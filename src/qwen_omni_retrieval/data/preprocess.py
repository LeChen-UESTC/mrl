from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from qwen_omni_retrieval.data.annotation import NormalizedRecord, load_annotations
from qwen_omni_retrieval.data.media_paths import resolve_media_paths
from qwen_omni_retrieval.data.modality import TEXT_MODALITIES, validate_modalities
from qwen_omni_retrieval.data.serialization import encode_jsonable


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

RAW_MEDIA_KEYS = {"input_features", "pixel_values", "pixel_values_videos"}
MEDIA_FEATURE_KEYS = {"audio_features", "image_features", "video_features"}
MEDIA_MODALITIES = {"audio", "video"}
CACHE_FORMAT = "encoder_features_v1"


def normalize_nframes(value: Any) -> int | None:
    if value is None or value == "":
        return None
    nframes = int(value)
    if nframes <= 0:
        raise ValueError(f"nframes must be a positive integer when set, got {nframes}.")
    return nframes


def normalize_optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None or value == "":
        return None
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer when set, got {number}.")
    return number


def normalize_log_every(value: Any) -> int:
    if value is None or value == "":
        return 100
    number = int(value)
    if number < 0:
        raise ValueError(f"log_every must be non-negative, got {number}.")
    return number


def normalize_bool(value: Any, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def normalize_feature_dtype(value: Any) -> torch.dtype:
    if value is None or value == "":
        return torch.float16
    normalized = str(value).strip().lower()
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported feature_dtype: {value}. Use fp16, bf16, or fp32.")


def feature_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype).replace("torch.", "")


def output_dir_with_frame_suffix(output_dir: str | Path, nframes: int | None) -> Path:
    output_path = Path(output_dir)
    if nframes is None:
        return output_path
    suffix = f"_n_frames_{nframes}"
    if output_path.name.endswith(suffix):
        return output_path
    return output_path.with_name(f"{output_path.name}{suffix}")


def build_messages(
    modality: str,
    payload: str,
    *,
    video_nframes: int | None = None,
) -> list[dict[str, Any]]:
    if modality in TEXT_MODALITIES:
        content = [
            {"type": "text", "text": payload},
            {"type": "text", "text": "Conclude above text in one word:"},
        ]
    elif modality == "video":
        video_item: dict[str, Any] = {"type": "video", "video": payload}
        if video_nframes is not None:
            video_item["nframes"] = video_nframes
        content = [
            video_item,
            {"type": "text", "text": "Conclude above video in one word:"},
        ]
    elif modality == "audio":
        content = [
            {"type": "audio", "audio": payload},
            {"type": "text", "text": "Conclude above audio in one word:"},
        ]
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def processor_inputs_for_messages(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    use_audio_in_video: bool,
) -> dict[str, torch.Tensor]:
    from qwen_omni_utils import process_mm_info

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios if audios else None,
        images=images if images else None,
        videos=videos if videos else None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    return {key: value.cpu() for key, value in inputs.items() if torch.is_tensor(value)}


def move_inputs_to_model_device(inputs: dict[str, torch.Tensor], model: Any) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    return {key: value.to(device) for key, value in inputs.items()}


def load_media_feature_model(config: dict[str, Any]) -> Any:
    from transformers import AutoConfig, Qwen2_5OmniThinkerForConditionalGeneration

    model_cfg = config.get("model", {})
    processor_cfg = config["processor"]
    model_path = model_cfg.get("model_path") or processor_cfg.get("model_path")
    if not model_path:
        raise ValueError(
            "cache.cache_media_features=true requires model.model_path in the preprocess config."
        )
    local_files_only = model_cfg.get(
        "local_files_only",
        processor_cfg.get("local_files_only", True),
    )
    attn_implementation = model_cfg.get("attn_implementation", "sdpa")
    model_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    thinker_config = getattr(model_config, "thinker_config", model_config)
    load_kwargs: dict[str, Any] = {
        "config": thinker_config,
        "torch_dtype": model_cfg.get("torch_dtype", "auto"),
        "trust_remote_code": True,
        "local_files_only": local_files_only,
        "attn_implementation": attn_implementation,
    }
    device_map = model_cfg.get("device_map", "auto")
    if device_map not in {None, "", "none", "None"}:
        load_kwargs["device_map"] = device_map
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    if "device_map" not in load_kwargs:
        device_name = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        thinker.to(torch.device(device_name))
    thinker.eval()
    for param in thinker.parameters():
        param.requires_grad_(False)
    return thinker


@torch.inference_mode()
def extract_media_features(
    thinker: Any,
    inputs: dict[str, torch.Tensor],
    *,
    feature_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    model_inputs = move_inputs_to_model_device(inputs, thinker)
    features: dict[str, torch.Tensor] = {}
    if model_inputs.get("input_features") is not None:
        audio_features = thinker.get_audio_features(
            input_features=model_inputs["input_features"],
            feature_attention_mask=model_inputs.get("feature_attention_mask"),
            return_dict=True,
        ).last_hidden_state
        features["audio_features"] = audio_features.detach().to(dtype=feature_dtype, device="cpu")

    if model_inputs.get("pixel_values") is not None:
        image_features = thinker.get_image_features(
            model_inputs["pixel_values"],
            model_inputs["image_grid_thw"],
            return_dict=True,
        ).pooler_output
        features["image_features"] = image_features.detach().to(dtype=feature_dtype, device="cpu")

    if model_inputs.get("pixel_values_videos") is not None:
        video_features = thinker.get_video_features(
            model_inputs["pixel_values_videos"],
            model_inputs["video_grid_thw"],
            return_dict=True,
        ).pooler_output
        features["video_features"] = video_features.detach().to(dtype=feature_dtype, device="cpu")
    return features


def split_cache_inputs(
    inputs: dict[str, torch.Tensor],
    *,
    features: dict[str, torch.Tensor],
    cache_raw_processor_tensors: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    token_inputs = {
        key: value.cpu()
        for key, value in inputs.items()
        if key not in RAW_MEDIA_KEYS and key not in MEDIA_FEATURE_KEYS
    }
    media_inputs = dict(features)
    if cache_raw_processor_tensors:
        media_inputs.update({key: value.cpu() for key, value in inputs.items() if key in RAW_MEDIA_KEYS})
    return token_inputs, media_inputs


def record_payloads(
    record: NormalizedRecord,
    *,
    video_path: str,
    audio_path: str | None,
) -> dict[str, list[str] | str]:
    payloads: dict[str, list[str] | str] = {
        "vision_cap": record.vision_caps,
        "video": video_path,
    }
    if audio_path:
        payloads["audio"] = audio_path
    if record.subtitle:
        payloads["subtitle"] = record.subtitle
    if record.vast_cap:
        payloads["vast_cap"] = record.vast_cap
    return payloads


def build_progress_bar(records: list[NormalizedRecord], *, dataset_name: str) -> tuple[Any, Any | None]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return enumerate(records), None
    pbar = tqdm(
        enumerate(records),
        total=len(records),
        desc=f"preprocess {dataset_name or 'dataset'}",
        dynamic_ncols=True,
    )
    return pbar, pbar


def preprocess_dataset(config: dict[str, Any]) -> dict[str, Any]:
    from transformers import Qwen2_5OmniProcessor

    dataset_cfg = config["dataset"]
    processor_cfg = config["processor"]
    cache_cfg = config["cache"]
    video_cfg = config.get("video", {})
    video_nframes = normalize_nframes(video_cfg.get("nframes"))
    max_samples = normalize_optional_positive_int(cache_cfg.get("max_samples"), name="max_samples")
    log_every = normalize_log_every(cache_cfg.get("log_every", 100))
    cache_media_features = normalize_bool(cache_cfg.get("cache_media_features"), default=True)
    cache_raw_processor_tensors = normalize_bool(
        cache_cfg.get("cache_raw_processor_tensors"),
        default=False,
    )
    feature_dtype = normalize_feature_dtype(cache_cfg.get("feature_dtype", "fp16"))

    dataset_name = dataset_cfg.get("name", "")
    allow_vast_cap = dataset_name.lower() == "vast"
    modalities = list(cache_cfg.get("modalities_to_cache", ["vision_cap", "video"]))
    required_modalities = list(cache_cfg.get("required_modalities", ["vision_cap", "video"]))
    validate_modalities(modalities, dataset_name=dataset_name, allow_vast_cap=allow_vast_cap)
    validate_modalities(required_modalities, dataset_name=dataset_name, allow_vast_cap=allow_vast_cap)
    if not cache_media_features and not cache_raw_processor_tensors and set(modalities) & MEDIA_MODALITIES:
        raise ValueError(
            "At least one of cache.cache_media_features or cache.cache_raw_processor_tensors "
            "must be true when caching video/audio modalities."
        )

    output_dir = output_dir_with_frame_suffix(cache_cfg["output_dir"], video_nframes)
    token_dir = output_dir / "text_tokens"
    feature_dir = output_dir / "feature_shards"
    token_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    processor = Qwen2_5OmniProcessor.from_pretrained(
        processor_cfg["processor_path"],
        trust_remote_code=True,
        local_files_only=processor_cfg.get("local_files_only", True),
    )
    media_feature_model = (
        load_media_feature_model(config)
        if cache_media_features and bool(set(modalities) & MEDIA_MODALITIES)
        else None
    )

    all_records = load_annotations(dataset_cfg["anno_path"])
    records = all_records[:max_samples] if max_samples is not None else all_records
    shard_size = int(cache_cfg.get("shard_size", 128))
    use_audio_in_video = bool(dataset_cfg.get("use_audio_in_video", False))
    audio_from_video_if_missing = bool(dataset_cfg.get("audio_from_video_if_missing", True))

    manifest_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    feature_shard: dict[str, dict[str, Any]] = {}
    shard_idx = 0
    skipped: list[dict[str, str]] = []

    def flush_shard() -> None:
        nonlocal token_rows, feature_shard, shard_idx
        if not token_rows:
            return
        token_name = f"shard_{shard_idx:05d}.jsonl"
        with (token_dir / token_name).open("w", encoding="utf-8") as token_f:
            for token_row in token_rows:
                token_f.write(json.dumps(token_row, ensure_ascii=False) + "\n")
        if feature_shard:
            feature_name = f"shard_{shard_idx:05d}.pt"
            torch.save(feature_shard, feature_dir / feature_name)
        token_rows = []
        feature_shard = {}
        shard_idx += 1

    progress_iter, progress_bar = build_progress_bar(records, dataset_name=dataset_name)

    def report_progress(processed: int) -> None:
        stats = {
            "cached": len(manifest_rows),
            "skipped": len(skipped),
            "shard": shard_idx,
        }
        if progress_bar is not None:
            progress_bar.set_postfix(stats)
        elif log_every > 0 and (processed == 1 or processed % log_every == 0 or processed == len(records)):
            print(
                "preprocess "
                f"{dataset_name}: processed={processed}/{len(records)} "
                f"cached={stats['cached']} skipped={stats['skipped']} shard={stats['shard']}",
                flush=True,
            )

    try:
        with manifest_path.open("w", encoding="utf-8") as manifest_f:
            for idx, record in progress_iter:
                processed = idx + 1
                media = resolve_media_paths(
                    record.video_id,
                    dataset_cfg["video_dir"],
                    audio_dir=dataset_cfg.get("audio_dir"),
                    use_audio_in_video=use_audio_in_video,
                    audio_from_video_if_missing=audio_from_video_if_missing,
                )
                payloads = record_payloads(
                    record,
                    video_path=media.video_path,
                    audio_path=media.audio_path,
                )
                missing_required = [mod for mod in required_modalities if mod not in payloads]
                if missing_required:
                    skipped.append({"video_id": record.video_id, "reason": f"missing required {missing_required}"})
                    report_progress(processed)
                    continue

                cache_key = f"{record.video_id}::{idx}"
                token_item: dict[str, Any] = {}
                media_item: dict[str, Any] = {}
                ok = True
                for modality in modalities:
                    if modality not in payloads:
                        continue
                    payload = payloads[modality]
                    try:
                        if modality == "vision_cap":
                            token_item[modality] = [
                                processor_inputs_for_messages(
                                    processor,
                                    build_messages(modality, caption),
                                    use_audio_in_video=False,
                                )
                                for caption in payload
                            ]
                        else:
                            modality_use_audio = use_audio_in_video if modality == "video" else False
                            inputs = processor_inputs_for_messages(
                                processor,
                                build_messages(
                                    modality,
                                    str(payload),
                                    video_nframes=video_nframes if modality == "video" else None,
                                ),
                                use_audio_in_video=modality_use_audio,
                            )
                            features = (
                                extract_media_features(
                                    media_feature_model,
                                    inputs,
                                    feature_dtype=feature_dtype,
                                )
                                if media_feature_model is not None and modality in MEDIA_MODALITIES
                                else {}
                            )
                            modality_tokens, modality_media = split_cache_inputs(
                                inputs,
                                features=features,
                                cache_raw_processor_tensors=cache_raw_processor_tensors,
                            )
                            if modality in MEDIA_MODALITIES and not modality_media:
                                raise ValueError(
                                    f"{modality} produced no cached media features or raw media tensors."
                                )
                            token_item[modality] = modality_tokens
                            if modality_media:
                                media_item[modality] = modality_media
                    except Exception as exc:
                        skipped.append({"video_id": record.video_id, "reason": f"{modality}: {exc}"})
                        if modality in required_modalities:
                            ok = False
                            break
                        continue
                if not ok:
                    report_progress(processed)
                    continue

                token_name = f"shard_{shard_idx:05d}.jsonl"
                feature_name = f"shard_{shard_idx:05d}.pt"
                token_rows.append(
                    {
                        "cache_key": cache_key,
                        "modalities": encode_jsonable(token_item),
                    }
                )
                if media_item:
                    feature_shard[cache_key] = media_item
                row = {
                    "cache_format": CACHE_FORMAT,
                    "sample_id": record.sample_id,
                    "video_id": record.video_id,
                    "caption_count": len(record.vision_caps),
                    "video_path": media.video_path,
                    "audio_path": media.audio_path,
                    "use_audio_in_video": media.use_audio_in_video,
                    "nframes": video_nframes,
                    "available_modalities": sorted(token_item.keys()),
                    "token_shard": f"text_tokens/{token_name}",
                    "feature_shard": f"feature_shards/{feature_name}" if media_item else None,
                    "cache_key": cache_key,
                }
                manifest_rows.append(row)
                manifest_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                if len(token_rows) >= shard_size:
                    flush_shard()
                report_progress(processed)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    flush_shard()
    summary = {
        "dataset": dataset_name,
        "records_loaded": len(all_records),
        "records_selected": len(records),
        "records_cached": len(manifest_rows),
        "records_skipped": len(skipped),
        "max_samples": max_samples,
        "log_every": log_every,
        "cache_format": CACHE_FORMAT,
        "cache_media_features": cache_media_features,
        "cache_raw_processor_tensors": cache_raw_processor_tensors,
        "feature_dtype": feature_dtype_name(feature_dtype),
        "modalities_to_cache": modalities,
        "required_modalities": required_modalities,
        "nframes": video_nframes,
        "manifest_path": str(manifest_path),
        "shard_count": shard_idx,
        "skipped_examples": skipped[:20],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

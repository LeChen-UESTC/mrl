from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from qwen_omni_retrieval.data.annotation import NormalizedRecord, load_annotations
from qwen_omni_retrieval.data.media_paths import resolve_media_paths
from qwen_omni_retrieval.data.modality import TEXT_MODALITIES, validate_modalities


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def build_messages(modality: str, payload: str) -> list[dict[str, Any]]:
    if modality in TEXT_MODALITIES:
        content = [
            {"type": "text", "text": payload},
            {"type": "text", "text": "Conclude above text in one word:"},
        ]
    elif modality == "video":
        content = [
            {"type": "video", "video": payload},
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


def preprocess_dataset(config: dict[str, Any]) -> dict[str, Any]:
    from transformers import Qwen2_5OmniProcessor

    dataset_cfg = config["dataset"]
    processor_cfg = config["processor"]
    cache_cfg = config["cache"]

    dataset_name = dataset_cfg.get("name", "")
    allow_vast_cap = dataset_name.lower() == "vast"
    modalities = list(cache_cfg.get("modalities_to_cache", ["vision_cap", "video"]))
    required_modalities = list(cache_cfg.get("required_modalities", ["vision_cap", "video"]))
    validate_modalities(modalities, dataset_name=dataset_name, allow_vast_cap=allow_vast_cap)
    validate_modalities(required_modalities, dataset_name=dataset_name, allow_vast_cap=allow_vast_cap)

    output_dir = Path(cache_cfg["output_dir"])
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    processor = Qwen2_5OmniProcessor.from_pretrained(
        processor_cfg["processor_path"],
        trust_remote_code=True,
        local_files_only=processor_cfg.get("local_files_only", True),
    )

    records = load_annotations(dataset_cfg["anno_path"])
    shard_size = int(cache_cfg.get("shard_size", 128))
    use_audio_in_video = bool(dataset_cfg.get("use_audio_in_video", False))
    audio_from_video_if_missing = bool(dataset_cfg.get("audio_from_video_if_missing", True))

    manifest_rows: list[dict[str, Any]] = []
    shard: dict[str, dict[str, Any]] = {}
    shard_idx = 0
    skipped: list[dict[str, str]] = []

    def flush_shard() -> None:
        nonlocal shard, shard_idx
        if not shard:
            return
        shard_name = f"shard_{shard_idx:05d}.pt"
        torch.save(shard, shard_dir / shard_name)
        shard = {}
        shard_idx += 1

    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        for idx, record in enumerate(records):
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
                continue

            cache_key = f"{record.video_id}::{idx}"
            item: dict[str, Any] = {}
            ok = True
            for modality in modalities:
                if modality not in payloads:
                    continue
                payload = payloads[modality]
                try:
                    if modality == "vision_cap":
                        item[modality] = [
                            processor_inputs_for_messages(
                                processor,
                                build_messages(modality, caption),
                                use_audio_in_video=False,
                            )
                            for caption in payload
                        ]
                    else:
                        modality_use_audio = use_audio_in_video if modality == "video" else False
                        item[modality] = processor_inputs_for_messages(
                            processor,
                            build_messages(modality, str(payload)),
                            use_audio_in_video=modality_use_audio,
                        )
                except Exception as exc:
                    skipped.append({"video_id": record.video_id, "reason": f"{modality}: {exc}"})
                    if modality in required_modalities:
                        ok = False
                        break
                    continue
            if not ok:
                continue

            shard[cache_key] = item
            shard_name = f"shard_{shard_idx:05d}.pt"
            row = {
                "sample_id": record.sample_id,
                "video_id": record.video_id,
                "caption_count": len(record.vision_caps),
                "video_path": media.video_path,
                "audio_path": media.audio_path,
                "use_audio_in_video": media.use_audio_in_video,
                "available_modalities": sorted(item.keys()),
                "cache_shard": f"shards/{shard_name}",
                "cache_key": cache_key,
            }
            manifest_rows.append(row)
            manifest_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if len(shard) >= shard_size:
                flush_shard()

    flush_shard()
    summary = {
        "dataset": dataset_name,
        "records_loaded": len(records),
        "records_cached": len(manifest_rows),
        "records_skipped": len(skipped),
        "modalities_to_cache": modalities,
        "required_modalities": required_modalities,
        "manifest_path": str(manifest_path),
        "shard_count": shard_idx,
        "skipped_examples": skipped[:20],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

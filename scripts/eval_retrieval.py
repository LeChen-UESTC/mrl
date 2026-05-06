#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_omni_retrieval.data.cache_dataset import CachedRetrievalDataset
from qwen_omni_retrieval.data.collator import QwenOmniCachedCollator
from qwen_omni_retrieval.data.modality import (
    parse_modalities,
    required_eval_modalities,
    validate_modalities,
)
from qwen_omni_retrieval.data.raw_dataset import (
    RawRetrievalCollator,
    RawRetrievalDataset,
    raw_batch_to_model_inputs,
)
from qwen_omni_retrieval.data.sampler import DistributedEvalSampler
from qwen_omni_retrieval.evaluation.retrieval import evaluate_retrieval_from_embeddings
from qwen_omni_retrieval.losses.contrastive import resolve_loss_mode
from qwen_omni_retrieval.models.projection import ProjectionHead
from qwen_omni_retrieval.models.qwen_thinker import infer_hidden_size, load_qwen_thinker_and_processor
from qwen_omni_retrieval.models.retrieval_model import QwenOmniRetrievalModel
from qwen_omni_retrieval.utils.config import load_config, save_json
from qwen_omni_retrieval.utils.distributed import (
    all_gather_object,
    cleanup_distributed,
    is_distributed,
    is_main_process,
    setup_distributed,
)
from qwen_omni_retrieval.utils.tensor import move_to_device


ALL_HEAD_MODALITIES = ["vision_cap", "video", "audio", "subtitle", "vast_cap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-Omni Gram-volume retrieval.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--query")
    parser.add_argument("--target")
    parser.add_argument("--aux", default=None)
    parser.add_argument("--loss_mode", choices=["inverse_volume", "neg_log", "cosine"], default=None)
    parser.add_argument("--output_json")
    return parser.parse_args()


def pad_token_id(processor: Any) -> int:
    tokenizer = getattr(processor, "tokenizer", None)
    value = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
    return 0 if value is None else int(value)


def optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None or value == "":
        return None
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive when set, got {number}.")
    return number


def eval_video_nframes(eval_cfg: dict[str, Any]) -> int | None:
    if "nframes" in eval_cfg:
        return optional_positive_int(eval_cfg.get("nframes"), name="eval nframes")
    return optional_positive_int(eval_cfg.get("video", {}).get("nframes"), name="eval video.nframes")


def load_eval_model(cfg: dict[str, Any], device: torch.device) -> tuple[QwenOmniRetrievalModel, Any]:
    thinker, processor = load_qwen_thinker_and_processor(cfg["model"])
    checkpoint_dir = cfg["eval"].get("checkpoint_dir")
    if checkpoint_dir:
        adapter_dir = Path(checkpoint_dir) / "adapter"
        if adapter_dir.exists():
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError("Loading a LoRA adapter requires `peft` in the existing environment.") from exc
            thinker = PeftModel.from_pretrained(thinker, adapter_dir, is_trainable=False)

    hidden_size = infer_hidden_size(thinker)
    projection_cfg = cfg.get("projection", {})
    projection = ProjectionHead(
        mode=projection_cfg.get("mode", "shared"),
        hidden_size=hidden_size,
        embed_dim=projection_cfg.get("embed_dim"),
        modalities=ALL_HEAD_MODALITIES,
        normalize=projection_cfg.get("normalize", True),
    )
    if checkpoint_dir:
        projection_path = Path(checkpoint_dir) / "projection_head.pt"
        if projection_path.exists():
            projection.load_state_dict(torch.load(projection_path, map_location="cpu"))

    model = QwenOmniRetrievalModel(
        thinker=thinker,
        projection=projection,
        use_audio_in_video_by_modality={
            "video": bool(cfg["eval"].get("use_audio_in_video", False)),
            "audio": False,
        },
    )
    model.to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def run_eval(cfg: dict[str, Any], model: Any, processor: Any, device: torch.device) -> dict[str, float]:
    eval_cfg = cfg["eval"]
    dataset_name = eval_cfg.get("name", "eval")
    query_modality = eval_cfg["query_modality"]
    target_modality = eval_cfg["target_modality"]
    auxiliary_modalities = parse_modalities(eval_cfg.get("auxiliary_modalities", []))
    loss_mode = resolve_loss_mode(cfg.get("loss", {}))
    if loss_mode == "cosine":
        auxiliary_modalities = []
    required = required_eval_modalities(query_modality, target_modality, auxiliary_modalities)
    validate_modalities(required, dataset_name=dataset_name, allow_vast_cap=dataset_name.lower() == "vast")

    dataset_source = "cache"
    raw_mode = False
    try:
        dataset = CachedRetrievalDataset(
            eval_cfg["cache_dir"],
            required_modalities=required,
            caption_selection=eval_cfg.get("caption_selection", "random"),
        )
        collate_fn = QwenOmniCachedCollator(modalities=required, pad_token_id=pad_token_id(processor))
    except (FileNotFoundError, ValueError) as exc:
        if not eval_cfg.get("anno_path") or not eval_cfg.get("video_dir"):
            raise
        dataset_source = "raw"
        raw_mode = True
        if is_main_process():
            print(
                {
                    "eval_fallback": dataset_name,
                    "source": "raw",
                    "reason": str(exc),
                },
                flush=True,
            )
        dataset = RawRetrievalDataset(
            anno_path=eval_cfg["anno_path"],
            video_dir=eval_cfg["video_dir"],
            audio_dir=eval_cfg.get("audio_dir"),
            required_modalities=required,
            use_audio_in_video=bool(eval_cfg.get("use_audio_in_video", False)),
            audio_from_video_if_missing=bool(eval_cfg.get("audio_from_video_if_missing", True)),
            caption_selection=eval_cfg.get("caption_selection", "random"),
        )
        collate_fn = RawRetrievalCollator()

    sampler = DistributedEvalSampler(len(dataset)) if is_distributed() else None
    loader = DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", 2)),
        shuffle=False,
        sampler=sampler,
        num_workers=int(eval_cfg.get("num_workers", 2)),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    core = model.module if hasattr(model, "module") else model
    previous_video_use_audio = core.use_audio_in_video_by_modality.get("video", False)
    core.use_audio_in_video_by_modality["video"] = bool(
        eval_cfg.get("use_audio_in_video", previous_video_use_audio)
    )
    pad_id = pad_token_id(processor)
    local_records: list[dict[str, Any]] = []
    raw_skipped: list[dict[str, str]] = []
    try:
        for batch in loader:
            if raw_mode:
                raw_inputs, valid_positions, skipped = raw_batch_to_model_inputs(
                    batch,
                    required_modalities=required,
                    processor=processor,
                    pad_token_id=pad_id,
                    video_nframes=eval_video_nframes(eval_cfg),
                )
                raw_skipped.extend(skipped)
                if not valid_positions:
                    continue
                modality_inputs = {
                    modality: move_to_device(inputs, device)
                    for modality, inputs in raw_inputs.items()
                }
                record_positions = valid_positions
            else:
                modality_inputs = {
                    modality: move_to_device(batch["modalities"][modality], device)
                    for modality in required
                }
                record_positions = list(range(len(batch["video_ids"])))

            embeddings = {
                modality: tensor.detach().float().cpu()
                for modality, tensor in model(modality_inputs).items()
            }
            for emb_idx, position in enumerate(record_positions):
                video_id = batch["video_ids"][position]
                local_records.append(
                    {
                        "index": int(batch["indices"][position]),
                        "video_id": video_id,
                        "query": embeddings[query_modality][emb_idx],
                        "target": embeddings[target_modality][emb_idx],
                        "aux": [embeddings[mod][emb_idx] for mod in auxiliary_modalities],
                    }
                )
    finally:
        core.use_audio_in_video_by_modality["video"] = previous_video_use_audio

    gathered = all_gather_object(local_records)
    gathered_skipped = all_gather_object(raw_skipped)
    if not is_main_process():
        return {}
    records = [item for part in gathered for item in part]
    if not records:
        raise ValueError(f"No valid eval records were produced for {dataset_name}.")
    records.sort(key=lambda item: item["index"])
    query_embeddings = torch.stack([item["query"] for item in records], dim=0)
    target_embeddings = torch.stack([item["target"] for item in records], dim=0)
    aux_embeddings = [
        torch.stack([item["aux"][idx] for item in records], dim=0)
        for idx in range(len(auxiliary_modalities))
    ]
    ids = [item["video_id"] for item in records]
    loss_cfg = cfg.get("loss", {})
    metrics = evaluate_retrieval_from_embeddings(
        query_embeddings=query_embeddings,
        target_embeddings=target_embeddings,
        auxiliary_embeddings=aux_embeddings,
        query_ids=ids,
        target_ids=ids,
        mode=loss_mode,
        score_mode=loss_cfg.get("score_mode", "inverse_volume"),
        scale=float(loss_cfg.get("volume_scale", 10.0)),
        temperature=float(loss_cfg.get("temperature", 1.0)),
        eps=float(loss_cfg.get("volume_eps", 1.0e-6)),
    )
    if dataset_source == "raw":
        metrics["raw_skipped"] = float(sum(len(items) for items in gathered_skipped))
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.checkpoint_dir:
        cfg["eval"]["checkpoint_dir"] = args.checkpoint_dir
    if args.query:
        cfg["eval"]["query_modality"] = args.query
    if args.target:
        cfg["eval"]["target_modality"] = args.target
    if args.aux is not None:
        cfg["eval"]["auxiliary_modalities"] = parse_modalities(args.aux)
    if args.loss_mode is not None:
        cfg.setdefault("loss", {})["mode"] = args.loss_mode
        cfg["loss"]["score_mode"] = args.loss_mode
    if args.output_json:
        cfg["eval"]["output_json"] = args.output_json

    device, _, _, _ = setup_distributed()
    try:
        model, processor = load_eval_model(cfg, device)
        if is_distributed():
            model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=True,
            )
        metrics = run_eval(cfg, model, processor, device)
        if is_main_process():
            print(metrics)
            output_json = cfg["eval"].get("output_json")
            if output_json:
                save_json(metrics, output_json)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

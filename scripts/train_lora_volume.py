#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_omni_retrieval.data.cache_dataset import CachedRetrievalDataset
from qwen_omni_retrieval.data.collator import QwenOmniCachedCollator
from qwen_omni_retrieval.data.modality import (
    normalize_train_modalities,
    parse_modalities,
    required_eval_modalities,
    required_train_modalities,
    validate_modalities,
    validate_train_extra_modalities,
)
from qwen_omni_retrieval.data.raw_dataset import (
    RawRetrievalCollator,
    RawRetrievalDataset,
    raw_batch_to_model_inputs,
)
from qwen_omni_retrieval.data.sampler import DistributedEvalSampler
from qwen_omni_retrieval.evaluation.retrieval import evaluate_retrieval_from_embeddings
from qwen_omni_retrieval.losses.contrastive import (
    resolve_loss_mode,
    resolve_score_mode,
    symmetric_contrastive_loss,
)
from qwen_omni_retrieval.models.lora import apply_lora, trainable_parameter_summary
from qwen_omni_retrieval.models.projection import ProjectionHead
from qwen_omni_retrieval.models.qwen_thinker import infer_hidden_size, load_qwen_thinker_and_processor
from qwen_omni_retrieval.models.retrieval_model import QwenOmniRetrievalModel
from qwen_omni_retrieval.utils.config import load_config, save_json
from qwen_omni_retrieval.utils.distributed import (
    all_gather_object,
    barrier,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    setup_distributed,
)
from qwen_omni_retrieval.utils.naming import DEFAULT_MODEL_OUTPUT_ROOT, training_run_name
from qwen_omni_retrieval.utils.seed import set_seed
from qwen_omni_retrieval.utils.tensor import move_to_device


ALL_HEAD_MODALITIES = ["vision_cap", "video", "audio", "subtitle", "vast_cap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP LoRA training with Gram-volume retrieval loss.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--modality", nargs="+", default=None)
    parser.add_argument("--extra_modalities", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--eval_nframes", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--log_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--do_eval", choices=["true", "false"], default=None)
    parser.add_argument("--loss_mode", choices=["inverse_volume", "neg_log", "cosine"], default=None)
    parser.add_argument("--wandb_mode", default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_target_modules", nargs="+", default=None)
    parser.add_argument("--lora_bias", default=None)
    return parser.parse_args()


def unwrap_ddp(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def pad_token_id(processor: Any) -> int:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return 0
    value = getattr(tokenizer, "pad_token_id", None)
    return 0 if value is None else int(value)


def str_to_bool(value: Any, *, default: bool) -> bool:
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


def resolve_training_modalities(training_cfg: dict[str, Any], *, dataset_name: str) -> list[str]:
    allow_vast_cap = dataset_name.lower() == "vast"
    if training_cfg.get("modalities") is not None:
        return normalize_train_modalities(
            parse_modalities(training_cfg.get("modalities")),
            dataset_name=dataset_name,
            allow_vast_cap=allow_vast_cap,
        )

    extra_modalities = parse_modalities(training_cfg.get("extra_modalities", []))
    validate_train_extra_modalities(
        extra_modalities,
        dataset_name=dataset_name,
        allow_vast_cap=allow_vast_cap,
    )
    if "vast_cap" in extra_modalities:
        raise ValueError(
            "Legacy `extra_modalities` cannot include `vast_cap` because `vision_cap` "
            "is the implicit text anchor. Use `modalities: [vast_cap, video, ...]` instead."
        )
    return normalize_train_modalities(
        required_train_modalities(extra_modalities),
        dataset_name=dataset_name,
        allow_vast_cap=allow_vast_cap,
    )


def training_text_anchor(modalities: list[str]) -> str:
    text_modalities = [modality for modality in modalities if modality in {"vision_cap", "vast_cap"}]
    if len(text_modalities) != 1:
        raise ValueError(f"Expected exactly one training text anchor, got {text_modalities}.")
    return text_modalities[0]


def training_loss_mode(modalities: list[str], loss_cfg: dict[str, Any]) -> str:
    if len(modalities) == 2:
        return "cosine"
    configured = resolve_loss_mode(loss_cfg)
    return configured if configured != "cosine" else "inverse_volume"


def configured_modalities_for_name(training_cfg: dict[str, Any], train_modalities: list[str]) -> list[str]:
    configured = parse_modalities(training_cfg.get("modalities"))
    return configured or train_modalities


def prepare_training_names(
    cfg: dict[str, Any],
    *,
    dataset_name: str,
    train_modalities: list[str],
) -> str:
    training_cfg = cfg.setdefault("training", {})
    run_name = training_run_name(
        dataset_name=dataset_name,
        loss_mode=training_loss_mode(train_modalities, cfg.get("loss", {})),
        modalities=configured_modalities_for_name(training_cfg, train_modalities),
        learning_rate=training_cfg.get("learning_rate", 1.0e-4),
        lora_cfg=cfg.get("lora", {}),
        projection_cfg=cfg.get("projection", {}),
    )
    output_root = Path(training_cfg.get("output_root", DEFAULT_MODEL_OUTPUT_ROOT))
    training_cfg["model_name"] = run_name
    training_cfg["output_root"] = str(output_root)
    training_cfg["output_dir"] = str(output_root / run_name)
    cfg.setdefault("wandb", {})["name"] = run_name
    return run_name


def set_cli_override(args: argparse.Namespace, section: dict[str, Any], arg_name: str, cfg_name: str | None = None) -> None:
    value = getattr(args, arg_name, None)
    if value is not None:
        section[cfg_name or arg_name] = value


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    training_cfg = cfg.setdefault("training", {})
    if args.modality is not None and args.extra_modalities is not None:
        raise ValueError("Use either `--modality` or legacy `--extra_modalities`, not both.")
    if args.modality is not None:
        training_cfg["modalities"] = parse_modalities(args.modality)
        training_cfg.pop("extra_modalities", None)
    elif args.extra_modalities is not None:
        training_cfg["extra_modalities"] = parse_modalities(args.extra_modalities)

    for arg_name in (
        "epochs",
        "max_steps",
        "batch_size",
        "eval_batch_size",
        "num_workers",
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "log_steps",
        "save_steps",
        "eval_steps",
    ):
        set_cli_override(args, training_cfg, arg_name)
    if args.do_eval is not None:
        training_cfg["do_eval"] = args.do_eval == "true"
    if args.eval_nframes is not None:
        for eval_cfg in cfg.get("eval_datasets", []):
            eval_cfg["nframes"] = args.eval_nframes

    if args.loss_mode is not None:
        loss_cfg = cfg.setdefault("loss", {})
        loss_cfg["mode"] = args.loss_mode
    if args.wandb_mode is not None:
        cfg.setdefault("wandb", {})["mode"] = args.wandb_mode

    lora_cfg = cfg.setdefault("lora", {})
    set_cli_override(args, lora_cfg, "lora_r", "r")
    set_cli_override(args, lora_cfg, "lora_alpha", "alpha")
    set_cli_override(args, lora_cfg, "lora_dropout", "dropout")
    set_cli_override(args, lora_cfg, "lora_bias", "bias")
    if args.lora_target_modules is not None:
        lora_cfg["target_modules"] = parse_modalities(args.lora_target_modules)


def total_train_steps(epochs: int, batches_per_epoch: int, max_steps: int) -> int:
    epoch_steps = max(0, epochs) * max(0, batches_per_epoch)
    if max_steps > 0:
        return min(max_steps, epoch_steps) if epoch_steps > 0 else max_steps
    return epoch_steps


def create_progress_bar(total: int) -> Any:
    if not is_main_process() or tqdm is None or total <= 0:
        return None
    return tqdm(total=total, desc="train", dynamic_ncols=True, leave=True)


def progress_postfix(train_log: dict[str, float]) -> dict[str, str]:
    postfix = {
        "loss": f"{train_log['train/loss']:.4g}",
        "lr": f"{train_log['train/lr']:.2e}",
    }
    for key in ("train/volume_mean", "train/logits_mean", "train/cosine_logits_mean"):
        if key in train_log:
            postfix[key.removeprefix("train/")] = f"{train_log[key]:.4g}"
    return postfix


def build_model(cfg: dict[str, Any], device: torch.device) -> tuple[QwenOmniRetrievalModel, Any]:
    thinker, processor = load_qwen_thinker_and_processor(cfg["model"])
    if cfg["training"].get("gradient_checkpointing", False) and hasattr(thinker, "gradient_checkpointing_enable"):
        thinker.gradient_checkpointing_enable()
    thinker = apply_lora(thinker, cfg["lora"])
    hidden_size = infer_hidden_size(thinker)
    projection_cfg = cfg.get("projection", {})
    projection = ProjectionHead(
        mode=projection_cfg.get("mode", "shared"),
        hidden_size=hidden_size,
        embed_dim=projection_cfg.get("embed_dim"),
        modalities=ALL_HEAD_MODALITIES,
        normalize=projection_cfg.get("normalize", True),
    )
    use_audio_in_video = bool(cfg["training"].get("use_audio_in_video", False))
    model = QwenOmniRetrievalModel(
        thinker=thinker,
        projection=projection,
        use_audio_in_video_by_modality={"video": use_audio_in_video, "audio": False},
    )
    model.to(device)
    return model, processor


def init_wandb(cfg: dict[str, Any]) -> Any:
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False) or not is_main_process():
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb logging is enabled but `wandb` is not installed in the current environment. "
            "This script will not install dependencies automatically."
        ) from exc
    return wandb.init(
        project=wandb_cfg.get("project", "qwen-omni-retrieval"),
        name=wandb_cfg.get("name"),
        mode=wandb_cfg.get("mode"),
        config=cfg,
    )


def make_dataloader(
    *,
    cache_dir: str,
    modalities: list[str],
    batch_size: int,
    pad_id: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    caption_selection: str,
) -> tuple[CachedRetrievalDataset, DataLoader, DistributedSampler | None]:
    dataset = CachedRetrievalDataset(
        cache_dir,
        required_modalities=modalities,
        caption_selection=caption_selection,
    )
    sampler = (
        DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        if is_distributed()
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=QwenOmniCachedCollator(modalities=modalities, pad_token_id=pad_id),
    )
    return dataset, loader, sampler


@torch.no_grad()
def evaluate_one_dataset(
    model: Any,
    eval_cfg: dict[str, Any],
    *,
    processor: Any,
    device: torch.device,
    pad_id: int,
    batch_size: int,
    num_workers: int,
    loss_cfg: dict[str, Any],
) -> dict[str, float]:
    core = unwrap_ddp(model)
    core.eval()

    dataset_name = eval_cfg.get("name", "eval")
    allow_vast_cap = dataset_name.lower() == "vast"
    query_modality = eval_cfg["query_modality"]
    target_modality = eval_cfg["target_modality"]
    auxiliary_modalities = parse_modalities(eval_cfg.get("auxiliary_modalities", []))
    if resolve_loss_mode(loss_cfg) == "cosine":
        auxiliary_modalities = []
    required = required_eval_modalities(query_modality, target_modality, auxiliary_modalities)
    validate_modalities(required, dataset_name=dataset_name, allow_vast_cap=allow_vast_cap)

    dataset_source = "cache"
    raw_mode = False
    try:
        dataset = CachedRetrievalDataset(
            eval_cfg["cache_dir"],
            required_modalities=required,
            caption_selection=eval_cfg.get("caption_selection", "random"),
        )
        collate_fn = QwenOmniCachedCollator(modalities=required, pad_token_id=pad_id)
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
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    previous_video_use_audio = core.use_audio_in_video_by_modality.get("video", False)
    core.use_audio_in_video_by_modality["video"] = bool(
        eval_cfg.get("use_audio_in_video", previous_video_use_audio)
    )
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
    records = [record for part in gathered for record in part]
    if not records:
        raise ValueError(f"No valid eval records were produced for {dataset_name}.")
    records.sort(key=lambda item: item["index"])
    query_embeddings = torch.stack([item["query"] for item in records], dim=0)
    target_embeddings = torch.stack([item["target"] for item in records], dim=0)
    aux_embeddings = []
    for aux_idx in range(len(auxiliary_modalities)):
        aux_embeddings.append(torch.stack([item["aux"][aux_idx] for item in records], dim=0))
    ids = [item["video_id"] for item in records]
    metrics = evaluate_retrieval_from_embeddings(
        query_embeddings=query_embeddings,
        target_embeddings=target_embeddings,
        auxiliary_embeddings=aux_embeddings,
        query_ids=ids,
        target_ids=ids,
        mode=resolve_loss_mode(loss_cfg),
        score_mode=resolve_score_mode(loss_cfg),
        scale=float(loss_cfg.get("volume_scale", 10.0)),
        temperature=float(loss_cfg.get("temperature", 1.0)),
        eps=float(loss_cfg.get("volume_eps", 1.0e-6)),
    )
    if dataset_source == "raw":
        metrics["raw_skipped"] = float(sum(len(items) for items in gathered_skipped))
    return metrics


def evaluate_all(
    model: Any,
    cfg: dict[str, Any],
    *,
    processor: Any,
    device: torch.device,
    pad_id: int,
    step: int,
    wandb_run: Any,
) -> dict[str, dict[str, float]]:
    eval_cfgs = cfg.get("eval_datasets", [])
    if not eval_cfgs:
        return {}
    training_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    results: dict[str, dict[str, float]] = {}
    for eval_cfg in eval_cfgs:
        metrics = evaluate_one_dataset(
            model,
            eval_cfg,
            processor=processor,
            device=device,
            pad_id=pad_id,
            batch_size=int(eval_cfg.get("batch_size", training_cfg.get("eval_batch_size", 2))),
            num_workers=int(eval_cfg.get("num_workers", training_cfg.get("num_workers", 2))),
            loss_cfg=loss_cfg,
        )
        barrier()
        if is_main_process():
            name = eval_cfg.get("name", "eval")
            results[name] = metrics
            if wandb_run is not None:
                wandb_run.log({f"eval/{name}/{k}": v for k, v in metrics.items()}, step=step)
    return results


def save_checkpoint(model: Any, cfg: dict[str, Any], *, step: int, metrics: dict[str, Any] | None = None) -> None:
    if not is_main_process():
        return
    core = unwrap_ddp(model)
    output_dir = Path(cfg["training"]["output_dir"])
    ckpt_dir = output_dir / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = ckpt_dir / "adapter"
    if hasattr(core.thinker, "save_pretrained"):
        core.thinker.save_pretrained(adapter_dir)
    else:
        torch.save(core.thinker.state_dict(), ckpt_dir / "thinker_state.pt")
    torch.save(core.projection.state_dict(), ckpt_dir / "projection_head.pt")
    save_json(cfg, ckpt_dir / "train_config.json")
    if metrics is not None:
        save_json(metrics, ckpt_dir / "eval_metrics.json")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)

    device, rank, world_size, _ = setup_distributed()
    progress_bar = None
    try:
        set_seed(int(cfg["training"].get("seed", 42)) + rank)
        dataset_name = cfg["training"].get("dataset_name", "vast")
        train_modalities = resolve_training_modalities(cfg["training"], dataset_name=dataset_name)
        text_anchor = training_text_anchor(train_modalities)
        model_name = prepare_training_names(
            cfg,
            dataset_name=dataset_name,
            train_modalities=train_modalities,
        )

        model, processor = build_model(cfg, device)
        if is_main_process():
            print(
                {
                    "model_name": model_name,
                    "output_dir": cfg["training"]["output_dir"],
                    "trainable_parameters": trainable_parameter_summary(model),
                    "train_modalities": train_modalities,
                    "train_loss_mode": training_loss_mode(train_modalities, cfg.get("loss", {})),
                }
            )
        if is_distributed():
            model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=True,
            )

        pad_id = pad_token_id(processor)
        training_cfg = cfg["training"]
        _, train_loader, train_sampler = make_dataloader(
            cache_dir=training_cfg["cache_dir"],
            modalities=train_modalities,
            batch_size=int(training_cfg.get("batch_size", 1)),
            pad_id=pad_id,
            shuffle=True,
            drop_last=True,
            num_workers=int(training_cfg.get("num_workers", 2)),
            caption_selection=training_cfg.get("caption_selection", "random"),
        )

        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(training_cfg.get("learning_rate", 1.0e-4)),
            weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        )
        wandb_run = init_wandb(cfg)
        output_dir = Path(training_cfg["output_dir"])
        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
            save_json(cfg, output_dir / "resolved_train_config.json")

        loss_cfg = cfg["loss"]
        epochs = int(training_cfg.get("epochs", 1))
        do_eval_enabled = str_to_bool(training_cfg.get("do_eval", True), default=True)
        eval_steps = int(training_cfg.get("eval_steps", 0))
        save_steps = int(training_cfg.get("save_steps", eval_steps if eval_steps > 0 else 0))
        grad_clip = float(training_cfg.get("max_grad_norm", 1.0))
        max_steps = int(training_cfg.get("max_steps", 0))
        global_step = 0
        log_steps = int(training_cfg.get("log_steps", 10))
        progress_bar = create_progress_bar(total_train_steps(epochs, len(train_loader), max_steps))

        for epoch in range(epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                model.train()
                modality_inputs = {
                    modality: move_to_device(batch["modalities"][modality], device)
                    for modality in train_modalities
                }
                embeddings = model(modality_inputs)

                train_loss_mode = training_loss_mode(train_modalities, loss_cfg)
                candidates = [embeddings[modality] for modality in train_modalities if modality != text_anchor]
                loss, stats = symmetric_contrastive_loss(
                    embeddings[text_anchor],
                    candidates,
                    mode=train_loss_mode,
                    scale=float(loss_cfg.get("volume_scale", 10.0)),
                    temperature=float(loss_cfg.get("temperature", 1.0)),
                    eps=float(loss_cfg.get("volume_eps", 1.0e-6)),
                    label_smoothing=float(loss_cfg.get("label_smoothing", 0.1)),
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [param for param in model.parameters() if param.requires_grad],
                        grad_clip,
                    )
                optimizer.step()
                global_step += 1

                if is_main_process():
                    train_log = {
                        "train/loss": float(loss.detach().cpu()),
                        "train/epoch": epoch,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                    train_log.update({f"train/{k}": float(v.cpu()) for k, v in stats.items()})
                    if wandb_run is not None:
                        wandb_run.log(train_log, step=global_step)
                    if progress_bar is not None:
                        progress_bar.set_postfix(progress_postfix(train_log))
                        progress_bar.update(1)
                    elif log_steps > 0 and global_step % log_steps == 0:
                        print({"step": global_step, **train_log})

                do_eval = do_eval_enabled and eval_steps > 0 and global_step % eval_steps == 0
                do_save = save_steps > 0 and global_step % save_steps == 0
                if do_eval:
                    metrics = evaluate_all(
                        model,
                        cfg,
                        processor=processor,
                        device=device,
                        pad_id=pad_id,
                        step=global_step,
                        wandb_run=wandb_run,
                    )
                    if do_save:
                        save_checkpoint(model, cfg, step=global_step, metrics=metrics)
                    barrier()
                elif do_save:
                    save_checkpoint(model, cfg, step=global_step)
                    barrier()

                if max_steps > 0 and global_step >= max_steps:
                    break
            if max_steps > 0 and global_step >= max_steps:
                break

        if progress_bar is not None:
            progress_bar.close()
            progress_bar = None

        final_metrics = (
            evaluate_all(
                model,
                cfg,
                processor=processor,
                device=device,
                pad_id=pad_id,
                step=global_step,
                wandb_run=wandb_run,
            )
            if do_eval_enabled
            else None
        )
        save_checkpoint(model, cfg, step=global_step, metrics=final_metrics)
        if wandb_run is not None:
            wandb_run.finish()
    finally:
        if progress_bar is not None:
            progress_bar.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()

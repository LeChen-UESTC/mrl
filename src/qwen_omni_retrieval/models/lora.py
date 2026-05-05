from __future__ import annotations

from typing import Any


def freeze_all_parameters(model: Any) -> None:
    for param in model.parameters():
        param.requires_grad = False


def apply_lora(thinker: Any, config: dict[str, Any]) -> Any:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA fine-tuning requires `peft` to already exist in this environment. "
            "Per your dependency constraint, this code will not install it automatically."
        ) from exc

    freeze_all_parameters(thinker)
    lora_config = LoraConfig(
        r=int(config.get("r", 16)),
        lora_alpha=int(config.get("alpha", 32)),
        lora_dropout=float(config.get("dropout", 0.05)),
        target_modules=list(config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        bias=config.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    thinker = get_peft_model(thinker, lora_config)
    return thinker


def trainable_parameter_summary(model: Any) -> dict[str, int | float]:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return {
        "trainable": trainable,
        "total": total,
        "ratio": (trainable / total) if total else 0.0,
    }

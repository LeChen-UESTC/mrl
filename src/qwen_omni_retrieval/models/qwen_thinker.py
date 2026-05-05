from __future__ import annotations

from typing import Any


def load_qwen_thinker_and_processor(config: dict[str, Any]) -> tuple[Any, Any]:
    from transformers import AutoConfig, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    model_path = config["model_path"]
    processor_path = config.get("processor_path", model_path)
    local_files_only = config.get("local_files_only", True)
    attn_implementation = config.get("attn_implementation", "sdpa")

    model_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    thinker_config = getattr(model_config, "thinker_config", model_config)
    processor = Qwen2_5OmniProcessor.from_pretrained(
        processor_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        config=thinker_config,
        torch_dtype=config.get("torch_dtype", "auto"),
        device_map=None,
        trust_remote_code=True,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    return thinker, processor


def infer_hidden_size(thinker: Any) -> int:
    cfg = getattr(thinker, "config", None)
    if cfg is not None:
        for name in ("hidden_size", "d_model"):
            value = getattr(cfg, name, None)
            if value:
                return int(value)
    embedding = thinker.get_input_embeddings()
    return int(embedding.weight.shape[1])

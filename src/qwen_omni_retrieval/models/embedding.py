from __future__ import annotations

from typing import Any

import torch


def unwrap_thinker(thinker: Any) -> Any:
    if hasattr(thinker, "module"):
        thinker = thinker.module
    base_model = getattr(thinker, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        return base_model.model
    return thinker


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_positions = torch.arange(attention_mask.size(1), device=hidden.device).unsqueeze(0)
    last_token_indices = (token_positions * attention_mask.long()).amax(dim=1)
    batch_indices = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_indices, last_token_indices]


def thinker_final_hidden_state(
    thinker: Any,
    inputs: dict[str, torch.Tensor],
    *,
    use_audio_in_video: bool,
) -> torch.Tensor:
    base = unwrap_thinker(thinker)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    inputs_embeds = base.get_input_embeddings()(input_ids)

    with torch.no_grad():
        if inputs.get("input_features") is not None:
            audio_features = base.get_audio_features(
                input_features=inputs["input_features"],
                feature_attention_mask=inputs.get("feature_attention_mask"),
                return_dict=True,
            ).last_hidden_state
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = base.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if inputs.get("pixel_values") is not None:
            image_features = base.get_image_features(
                inputs["pixel_values"],
                inputs["image_grid_thw"],
                return_dict=True,
            ).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, _ = base.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        if inputs.get("pixel_values_videos") is not None:
            video_features = base.get_video_features(
                inputs["pixel_values_videos"],
                inputs["video_grid_thw"],
                return_dict=True,
            ).pooler_output
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = base.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)

    feature_attention_mask = inputs.get("feature_attention_mask")
    audio_feature_lengths = (
        torch.sum(feature_attention_mask, dim=1) if feature_attention_mask is not None else None
    )

    position_ids = None
    if attention_mask is not None:
        position_ids, _ = base.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=attention_mask,
            use_audio_in_video=use_audio_in_video,
            audio_seqlens=audio_feature_lengths,
            second_per_grids=inputs.get("video_second_per_grid"),
        )

    outputs = base.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        return_dict=True,
    )
    return outputs.last_hidden_state

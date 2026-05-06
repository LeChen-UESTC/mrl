from __future__ import annotations

TEXT_MODALITIES = {"vision_cap", "subtitle", "vast_cap"}
MEDIA_MODALITIES = {"video", "audio"}
EXTRA_TRAIN_MODALITIES = {"audio", "subtitle", "vast_cap"}
ALL_MODALITIES = TEXT_MODALITIES | MEDIA_MODALITIES
TEXT_ANCHOR_MODALITIES = {"vision_cap", "vast_cap"}


def parse_modalities(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    raw_items = [value] if isinstance(value, str) else [str(item) for item in value]
    modalities: list[str] = []
    for item in raw_items:
        if not item.strip():
            continue
        parts = item.replace(";", ",").split(",")
        modalities.extend(part.strip() for part in parts if part.strip())
    return modalities


def validate_modalities(
    modalities: list[str],
    *,
    dataset_name: str | None = None,
    allow_vast_cap: bool = False,
) -> None:
    unknown = sorted(set(modalities) - ALL_MODALITIES)
    if unknown:
        raise ValueError(f"Unknown modalities: {unknown}. Supported: {sorted(ALL_MODALITIES)}")
    if "vast_cap" in modalities and not allow_vast_cap:
        ds = dataset_name or "this dataset"
        raise ValueError(f"`vast_cap` is only supported for VAST, but was requested for {ds}.")


def validate_train_extra_modalities(
    extra_modalities: list[str],
    *,
    dataset_name: str | None = None,
    allow_vast_cap: bool = False,
) -> None:
    unknown = sorted(set(extra_modalities) - EXTRA_TRAIN_MODALITIES)
    if unknown:
        raise ValueError(
            "Training extra modalities may only include "
            f"{sorted(EXTRA_TRAIN_MODALITIES)}, got {unknown}."
        )
    validate_modalities(extra_modalities, dataset_name=dataset_name, allow_vast_cap=allow_vast_cap)


def required_train_modalities(extra_modalities: list[str]) -> list[str]:
    return ["vision_cap", "video", *extra_modalities]


def normalize_train_modalities(
    modalities: list[str],
    *,
    dataset_name: str | None = None,
    allow_vast_cap: bool = False,
) -> list[str]:
    validate_modalities(modalities, dataset_name=dataset_name, allow_vast_cap=allow_vast_cap)
    if len(modalities) not in {2, 3, 4}:
        raise ValueError(f"Training requires 2, 3, or 4 modalities, got {len(modalities)}: {modalities}")
    if len(set(modalities)) != len(modalities):
        raise ValueError(f"Training modalities must not contain duplicates: {modalities}")

    text_anchors = [modality for modality in modalities if modality in TEXT_ANCHOR_MODALITIES]
    if len(text_anchors) != 1:
        raise ValueError(
            "Training modalities must contain exactly one text anchor: "
            "`vision_cap` or `vast_cap`."
        )
    if "video" not in modalities:
        raise ValueError("Training modalities must contain `video`.")

    text_anchor = text_anchors[0]
    auxiliary = [
        modality
        for modality in modalities
        if modality not in {text_anchor, "video"}
    ]
    return [text_anchor, "video", *auxiliary]


def required_eval_modalities(
    query_modality: str,
    target_modality: str,
    auxiliary_modalities: list[str],
) -> list[str]:
    modalities = [query_modality, target_modality, *auxiliary_modalities]
    seen: set[str] = set()
    ordered: list[str] = []
    for item in modalities:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered

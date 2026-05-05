from __future__ import annotations

TEXT_MODALITIES = {"vision_cap", "subtitle", "vast_cap"}
MEDIA_MODALITIES = {"video", "audio"}
EXTRA_TRAIN_MODALITIES = {"audio", "subtitle", "vast_cap"}
ALL_MODALITIES = TEXT_MODALITIES | MEDIA_MODALITIES


def parse_modalities(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        parts = value.replace(";", ",").split(",")
        return [part.strip() for part in parts if part.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


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

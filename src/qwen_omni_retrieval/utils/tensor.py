from __future__ import annotations

from typing import Any

import torch


def move_to_device(data: Any, device: torch.device) -> Any:
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    if isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [move_to_device(value, device) for value in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(value, device) for value in data)
    return data

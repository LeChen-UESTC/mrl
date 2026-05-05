from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> tuple[torch.device, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        return device, dist.get_rank(), dist.get_world_size(), local_rank
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, 0, 1, 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return tensor
    try:
        from torch.distributed.nn.functional import all_gather

        gathered = all_gather(tensor)
        return torch.cat(tuple(gathered), dim=0)
    except Exception:
        tensors = [torch.zeros_like(tensor) for _ in range(get_world_size())]
        dist.all_gather(tensors, tensor)
        tensors[get_rank()] = tensor
        return torch.cat(tensors, dim=0)


def all_gather_object(data: Any) -> list[Any]:
    if not is_distributed():
        return [data]
    gathered: list[Any] = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, data)
    return gathered

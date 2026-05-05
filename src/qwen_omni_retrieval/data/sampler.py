from __future__ import annotations

from collections.abc import Iterator

from torch.utils.data import Sampler

from qwen_omni_retrieval.utils.distributed import get_rank, get_world_size, is_distributed


class DistributedEvalSampler(Sampler[int]):
    """Shard eval indices across ranks without padding or duplicated samples."""

    def __init__(self, dataset_size: int) -> None:
        self.dataset_size = dataset_size
        self.rank = get_rank()
        self.world_size = get_world_size()

    def __iter__(self) -> Iterator[int]:
        if not is_distributed():
            yield from range(self.dataset_size)
            return
        yield from range(self.rank, self.dataset_size, self.world_size)

    def __len__(self) -> int:
        if not is_distributed():
            return self.dataset_size
        return (self.dataset_size + self.world_size - 1 - self.rank) // self.world_size

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(
        self,
        *,
        mode: str,
        hidden_size: int,
        embed_dim: int | None = None,
        modalities: list[str] | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.embed_dim = hidden_size if mode == "none" or embed_dim is None else embed_dim
        self.normalize = normalize

        if mode == "none":
            self.head = nn.Identity()
            self.heads = None
        elif mode == "shared":
            self.head = nn.Linear(hidden_size, self.embed_dim, bias=False)
            self.heads = None
        elif mode == "per_modality":
            if not modalities:
                raise ValueError("per_modality projection requires a modality list.")
            self.head = None
            self.heads = nn.ModuleDict(
                {modality: nn.Linear(hidden_size, self.embed_dim, bias=False) for modality in modalities}
            )
        else:
            raise ValueError("projection mode must be one of: none, shared, per_modality")

    def forward(self, hidden: torch.Tensor, modality: str) -> torch.Tensor:
        if self.mode == "per_modality":
            if self.heads is None or modality not in self.heads:
                raise KeyError(f"No projection head registered for modality `{modality}`.")
            out = self.heads[modality](hidden)
        else:
            out = self.head(hidden)
        if self.normalize:
            out = F.normalize(out, dim=-1)
        return out

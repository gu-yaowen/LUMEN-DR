from __future__ import annotations

import torch
from torch import nn


class BilinearPredictor(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Bilinear(hidden_dim, hidden_dim, 1, bias=True)

    def forward(
        self,
        drug_emb: torch.Tensor,
        disease_emb: torch.Tensor,
        drug_idx: torch.Tensor,
        disease_idx: torch.Tensor,
    ) -> torch.Tensor:
        d = drug_emb[drug_idx]
        s = disease_emb[disease_idx]
        return self.scorer(d, s).squeeze(-1)

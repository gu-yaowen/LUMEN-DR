from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import HGTConv


def activate_tensor(x: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "relu":
        return x.relu()
    if activation == "gelu":
        return torch.nn.functional.gelu(x)
    if activation == "leaky_relu":
        return torch.nn.functional.leaky_relu(x, negative_slope=0.1)
    raise ValueError(f"Unsupported activation: {activation}")


class HgtBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [HGTConv(in_channels=hidden_dim, out_channels=hidden_dim, metadata=metadata, heads=num_heads) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple[str, str, str], torch.Tensor]) -> dict[str, torch.Tensor]:
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
            x_dict = {k: self.dropout(activate_tensor(v, self.activation)) for k, v in x_dict.items()}
        return x_dict

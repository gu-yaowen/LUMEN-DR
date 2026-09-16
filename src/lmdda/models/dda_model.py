from __future__ import annotations

import torch
from torch import nn

from .heads import BilinearPredictor
from .hgt import HgtBackbone


def build_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "batchnorm":
        return nn.BatchNorm1d(dim)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def build_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    raise ValueError(f"Unsupported activation: {activation}")


class Projection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, norm_type: str, activation: str) -> None:
        super().__init__()
        self.net = nn.Sequential(
            build_norm(norm_type, in_dim),
            nn.Linear(in_dim, out_dim),
            build_activation(activation),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NodeEncoder(nn.Module):
    def __init__(
        self,
        source: str,
        in_dim: int | None,
        out_dim: int,
        dropout: float,
        norm_type: str,
        activation: str,
        num_nodes: int,
    ) -> None:
        super().__init__()
        self.source = source
        self.num_nodes = num_nodes
        if source == "pretrained":
            if in_dim is None:
                raise ValueError("pretrained node encoder requires input dimension")
            self.pretrained_proj = Projection(in_dim, out_dim, dropout, norm_type, activation)
            self.id_embedding = None
        elif source == "id":
            self.pretrained_proj = None
            self.id_embedding = nn.Embedding(num_nodes, out_dim)
        else:
            raise ValueError(f"Unsupported source: {source}")

    def forward(self, x: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        if self.source == "pretrained":
            if x is None:
                raise ValueError("pretrained node encoder expected tensor input")
            return self.pretrained_proj(x)
        node_idx = torch.arange(self.num_nodes, device=device, dtype=torch.long)
        return self.id_embedding(node_idx)


class LmddaModel(nn.Module):
    def __init__(
        self,
        feature_dims: dict[str, int | None],
        node_counts: dict[str, int],
        feature_sources: dict[str, str],
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        contrastive_dim: int,
        proj_norm_type: str,
        activation: str,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
    ) -> None:
        super().__init__()
        self.drug_encoder = NodeEncoder(
            feature_sources["drug"],
            feature_dims["drug"],
            hidden_dim,
            dropout,
            proj_norm_type,
            activation,
            node_counts["drug"],
        )
        self.disease_encoder = NodeEncoder(
            feature_sources["disease"],
            feature_dims["disease"],
            hidden_dim,
            dropout,
            proj_norm_type,
            activation,
            node_counts["disease"],
        )
        self.protein_encoder = NodeEncoder(
            feature_sources["protein"],
            feature_dims["protein"],
            hidden_dim,
            dropout,
            proj_norm_type,
            activation,
            node_counts["protein"],
        )
        self.backbone = HgtBackbone(hidden_dim, num_heads, num_layers, metadata, dropout, activation)
        self.predictor = BilinearPredictor(hidden_dim)
        self.drug_contrast_proj = nn.Linear(hidden_dim, contrastive_dim)
        self.disease_contrast_proj = nn.Linear(hidden_dim, contrastive_dim)

    def encode_nodes(
        self,
        feature_tensors: dict[str, torch.Tensor | None],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        x_dict = {
            "drug": self.drug_encoder(feature_tensors["drug"], device),
            "disease": self.disease_encoder(feature_tensors["disease"], device),
            "protein": self.protein_encoder(feature_tensors["protein"], device),
        }
        return self.backbone(x_dict, edge_index_dict)

    def forward(
        self,
        feature_tensors: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        drug_idx: torch.Tensor,
        disease_idx: torch.Tensor,
    ) -> torch.Tensor:
        x_dict = self.encode_nodes(feature_tensors, edge_index_dict)
        logits = self.predictor(x_dict["drug"], x_dict["disease"], drug_idx, disease_idx)
        return logits

    def score_pairs(
        self,
        x_dict: dict[str, torch.Tensor],
        drug_idx: torch.Tensor,
        disease_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.predictor(x_dict["drug"], x_dict["disease"], drug_idx, disease_idx)

    def contrastive_embeddings(
        self,
        x_dict: dict[str, torch.Tensor],
        drug_idx: torch.Tensor,
        disease_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        drug_repr = self.drug_contrast_proj(x_dict["drug"][drug_idx])
        disease_repr = self.disease_contrast_proj(x_dict["disease"][disease_idx])
        return drug_repr, disease_repr

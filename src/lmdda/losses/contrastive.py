from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_inbatch_infonce_loss(
    drug_repr: torch.Tensor,
    disease_repr: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if drug_repr.ndim != 2 or disease_repr.ndim != 2:
        raise ValueError("contrastive inputs must be rank-2 tensors")
    if drug_repr.shape != disease_repr.shape:
        raise ValueError("drug and disease contrastive tensors must have the same shape")
    if drug_repr.shape[0] < 2:
        return drug_repr.new_zeros(())

    drug_repr = F.normalize(drug_repr, p=2, dim=-1)
    disease_repr = F.normalize(disease_repr, p=2, dim=-1)
    logits = drug_repr @ disease_repr.T
    logits = logits / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_drug = F.cross_entropy(logits, labels)
    loss_disease = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_drug + loss_disease)

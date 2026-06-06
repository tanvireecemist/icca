from __future__ import annotations

import torch
from torch.nn import functional as F


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.new_zeros(())
    features = F.normalize(embeddings, dim=1)
    logits = features @ features.T / temperature
    identity = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    valid = positive.any(dim=1)
    if not valid.any():
        return embeddings.new_zeros(())

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~identity)
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    mean_positive = (positive * log_probability).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    return -mean_positive[valid].mean()


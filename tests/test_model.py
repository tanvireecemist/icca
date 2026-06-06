from __future__ import annotations

import torch

from rf_research.model.losses import supervised_contrastive_loss
from rf_research.model.network import DualDomainRFEncoder


def test_encoder_output_shape_and_finiteness() -> None:
    model = DualDomainRFEncoder(
        stem_width=16,
        model_dim=32,
        transformer_layers=1,
        attention_heads=4,
        embedding_dim=24,
        dropout=0.0,
    )
    output = model(torch.randn(3, 2, 512))
    assert output.shape == (3, 24)
    assert torch.isfinite(output).all()


def test_supervised_contrastive_loss_is_finite() -> None:
    embeddings = torch.randn(6, 16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = supervised_contrastive_loss(embeddings, labels, temperature=0.1)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss >= 0


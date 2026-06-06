from __future__ import annotations

import torch
from torch import nn

from rf_research.model.layers import SignalBranch, sinusoidal_position


class DualDomainRFEncoder(nn.Module):
    def __init__(
        self,
        stem_width: int,
        model_dim: int,
        transformer_layers: int,
        attention_heads: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.time_branch = SignalBranch(stem_width, model_dim)
        self.frequency_branch = SignalBranch(stem_width, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=attention_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_dim),
        )

    @staticmethod
    def spectral_view(iq: torch.Tensor) -> torch.Tensor:
        device_type = iq.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            value = torch.complex(iq[:, 0].float(), iq[:, 1].float())
            spectrum = torch.fft.fftshift(torch.fft.fft(value, dim=-1), dim=-1)
            magnitude = torch.log1p(torch.abs(spectrum))
            phase_step = torch.angle(spectrum[:, 1:] * spectrum[:, :-1].conj())
            phase_step = torch.nn.functional.pad(phase_step, (1, 0))
            features = torch.stack((magnitude, phase_step), dim=1)
            features = features - features.mean(dim=-1, keepdim=True)
            features = features / features.std(dim=-1, keepdim=True).clamp_min(1e-4)
        return features.to(iq.dtype)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        time_tokens = self.time_branch(iq)
        frequency_tokens = self.frequency_branch(self.spectral_view(iq))
        tokens = torch.cat((time_tokens, frequency_tokens), dim=1)
        position = sinusoidal_position(
            tokens.shape[1], tokens.shape[2], tokens.device
        ).to(tokens.dtype)
        tokens = self.transformer(tokens + position.unsqueeze(0))
        return self.output(tokens.mean(dim=1))


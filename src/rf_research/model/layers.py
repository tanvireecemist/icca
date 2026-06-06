from __future__ import annotations

import math

import torch
from torch import nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = weight
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.weight * gradient, None


def gradient_reverse(value: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(value, weight)


class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.net(value)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=9,
                stride=stride,
                padding=4,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            SqueezeExcitation1D(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(value) + self.skip(value))


class SignalBranch(nn.Module):
    def __init__(self, stem_width: int, model_dim: int) -> None:
        super().__init__()
        middle = min(model_dim, stem_width * 2)
        self.net = nn.Sequential(
            nn.Conv1d(2, stem_width, 15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(stem_width),
            nn.SiLU(),
            ResidualBlock1D(stem_width, stem_width, stride=2),
            ResidualBlock1D(stem_width, middle, stride=2),
            ResidualBlock1D(middle, model_dim, stride=2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).transpose(1, 2)


def sinusoidal_position(length: int, dimension: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / dimension)
    )
    encoding = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor)
    return encoding


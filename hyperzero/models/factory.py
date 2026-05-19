"""Model factory for v1 policy-value architectures."""

from __future__ import annotations

from typing import Literal

from torch import nn

from hyperzero.game.config import GameConfig
from hyperzero.models.cnn import PolicyValueCNN
from hyperzero.models.line_mlp import PolicyValueLineMLP
from hyperzero.models.mlp import PolicyValueMLP
from hyperzero.models.transformer import PolicyValueTransformer

ModelType = Literal[
    "mlp",
    "line_mlp",
    "cnn",
    "resnet",
    "transformer",
]


def build_policy_value_model(
    config: GameConfig,
    *,
    model_type: str = "mlp",
    hidden_size: int = 128,
    residual_blocks: int = 2,
) -> nn.Module:
    """Build a policy-value model sized for a game configuration."""
    if model_type == "mlp":
        return PolicyValueMLP.from_config(
            config,
            hidden_size=hidden_size,
            residual_blocks=residual_blocks,
        )
    if model_type == "line_mlp":
        return PolicyValueLineMLP.from_config(
            config,
            hidden_size=hidden_size,
            residual_blocks=residual_blocks,
        )
    if model_type == "cnn":
        return PolicyValueCNN.from_config(
            config,
            hidden_size=hidden_size,
            residual_blocks=0,
        )
    if model_type == "resnet":
        return PolicyValueCNN.from_config(
            config,
            hidden_size=hidden_size,
            residual_blocks=residual_blocks,
        )
    if model_type == "transformer":
        return PolicyValueTransformer.from_config(
            config,
            hidden_size=hidden_size,
            residual_blocks=residual_blocks,
        )
    raise ValueError(f"unknown model_type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return int(sum(parameter.numel() for parameter in model.parameters()))

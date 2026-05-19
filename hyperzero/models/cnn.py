"""CNN and ResNet-style policy-value networks for flat board inputs."""

from __future__ import annotations

import torch
from torch import nn

from hyperzero.game.config import GameConfig


def _conv_nd(rank: int):
    if rank == 2:
        return nn.Conv2d
    if rank == 3:
        return nn.Conv3d
    raise ValueError("cnn/resnet model_type supports only 2D and 3D boards")


def _adaptive_pool_nd(rank: int):
    if rank == 2:
        return nn.AdaptiveAvgPool2d
    if rank == 3:
        return nn.AdaptiveAvgPool3d
    raise ValueError("cnn/resnet model_type supports only 2D and 3D boards")


class _ResidualConvBlock(nn.Module):
    def __init__(self, rank: int, channels: int) -> None:
        super().__init__()
        conv = _conv_nd(rank)
        self.layers = nn.Sequential(
            conv(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            conv(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class PolicyValueCNN(nn.Module):
    """Small CNN/ResNet policy-value network for 2D or 3D boards."""

    def __init__(
        self,
        config: GameConfig,
        *,
        hidden_size: int = 128,
        residual_blocks: int = 2,
    ) -> None:
        super().__init__()
        if len(config.shape) not in (2, 3):
            raise ValueError("cnn/resnet model_type supports only 2D and 3D boards")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if residual_blocks < 0:
            raise ValueError("residual_blocks must be nonnegative")

        self.shape = config.shape
        self.rank = len(config.shape)
        self.num_cells = config.num_cells
        self.num_actions = config.num_actions
        self.hidden_size = int(hidden_size)
        self.residual_blocks = int(residual_blocks)
        conv = _conv_nd(self.rank)
        pool = _adaptive_pool_nd(self.rank)
        self.stem = nn.Sequential(
            conv(1, self.hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            *(
                _ResidualConvBlock(self.rank, self.hidden_size)
                for _ in range(self.residual_blocks)
            )
        )
        self.policy_head = nn.Sequential(
            conv(self.hidden_size, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2 * self.num_cells, self.num_actions),
        )
        self.value_head = nn.Sequential(
            pool(1),
            nn.Flatten(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Tanh(),
        )

    @classmethod
    def from_config(
        cls,
        config: GameConfig,
        *,
        hidden_size: int = 128,
        residual_blocks: int = 2,
    ) -> PolicyValueCNN:
        """Build a model sized for a game configuration."""
        return cls(
            config,
            hidden_size=hidden_size,
            residual_blocks=residual_blocks,
        )

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits and value for a batch or single board."""
        board = board.to(dtype=torch.float32)
        squeeze_batch = board.ndim == 1
        if squeeze_batch:
            board = board.unsqueeze(0)
        if board.ndim != 2 or board.shape[1] != self.num_cells:
            raise ValueError(
                f"board must have shape ({self.num_cells},) or batch x cells"
            )

        x = board.reshape((board.shape[0], 1, *self.shape))
        hidden = self.trunk(self.stem(x))
        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        if squeeze_batch:
            return policy_logits.squeeze(0), value.squeeze(0)
        return policy_logits, value

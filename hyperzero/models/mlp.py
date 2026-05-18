"""Minimal residual MLP policy-value network."""

from __future__ import annotations

import torch
from torch import nn

from hyperzero.game.config import GameConfig


class ResidualMLPBlock(nn.Module):
    """Two-layer residual block for fixed-width hidden vectors."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a residual MLP block."""
        return self.activation(x + self.layers(x))


class PolicyValueMLP(nn.Module):
    """Policy-value network over a flattened canonical board."""

    def __init__(
        self,
        num_cells: int,
        num_actions: int,
        *,
        hidden_size: int = 128,
        residual_blocks: int = 2,
    ) -> None:
        super().__init__()
        if num_cells <= 0:
            raise ValueError("num_cells must be positive")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if residual_blocks < 0:
            raise ValueError("residual_blocks must be nonnegative")

        self.num_cells = int(num_cells)
        self.num_actions = int(num_actions)
        self.hidden_size = int(hidden_size)
        self.residual_blocks = int(residual_blocks)

        self.input_layer = nn.Sequential(
            nn.Linear(self.num_cells, self.hidden_size),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            *(ResidualMLPBlock(self.hidden_size) for _ in range(self.residual_blocks))
        )
        self.policy_head = nn.Linear(self.hidden_size, self.num_actions)
        self.value_head = nn.Sequential(
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
    ) -> PolicyValueMLP:
        """Build a model sized for a game configuration."""
        return cls(
            config.num_cells,
            config.num_actions,
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

        hidden = self.trunk(self.input_layer(board))
        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        if squeeze_batch:
            return policy_logits.squeeze(0), value.squeeze(0)
        return policy_logits, value

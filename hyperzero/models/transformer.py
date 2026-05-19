"""Small board-token transformer policy-value network."""

from __future__ import annotations

import torch
from torch import nn

from hyperzero.game.config import GameConfig


class PolicyValueTransformer(nn.Module):
    """Transformer encoder over flattened board-cell tokens."""

    def __init__(
        self,
        config: GameConfig,
        *,
        hidden_size: int = 128,
        residual_blocks: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if residual_blocks <= 0:
            raise ValueError("transformer residual_blocks must be positive")
        if hidden_size % heads != 0:
            heads = 1

        self.num_cells = config.num_cells
        self.num_actions = config.num_actions
        self.hidden_size = int(hidden_size)
        self.residual_blocks = int(residual_blocks)
        self.token = nn.Linear(1, self.hidden_size)
        self.position = nn.Parameter(torch.zeros(1, self.num_cells, self.hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.residual_blocks)
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_cells * self.hidden_size, self.num_actions),
        )
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
    ) -> PolicyValueTransformer:
        """Build a model sized for a game configuration."""
        return cls(
            config,
            hidden_size=hidden_size,
            residual_blocks=max(1, residual_blocks),
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

        tokens = self.token(board.unsqueeze(-1)) + self.position
        encoded = self.encoder(tokens)
        policy_logits = self.policy_head(encoded)
        value = self.value_head(encoded.mean(dim=1)).squeeze(-1)
        if squeeze_batch:
            return policy_logits.squeeze(0), value.squeeze(0)
        return policy_logits, value

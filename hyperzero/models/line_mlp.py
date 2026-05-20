"""Line-aware MLP policy-value network."""

from __future__ import annotations

import torch
from torch import nn

from hyperzero.game.config import GameConfig
from hyperzero.models.mlp import ResidualMLPBlock


class PolicyValueLineMLP(nn.Module):
    """MLP over board plus differentiable open-line feature channels."""

    def __init__(
        self,
        config: GameConfig,
        *,
        hidden_size: int = 128,
        residual_blocks: int = 2,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if residual_blocks < 0:
            raise ValueError("residual_blocks must be nonnegative")

        self.num_cells = config.num_cells
        self.num_actions = config.num_actions
        self.hidden_size = int(hidden_size)
        self.residual_blocks = int(residual_blocks)
        lines = torch.as_tensor(config.winning_lines, dtype=torch.long)
        line_cell = torch.zeros((len(lines), config.num_cells), dtype=torch.float32)
        for line_index, line in enumerate(lines):
            line_cell[line_index, line] = 1.0
        weights = torch.as_tensor(
            [0.0, *[float(10**count) for count in range(1, config.connect_k + 1)]],
            dtype=torch.float32,
        )
        self.register_buffer("lines", lines, persistent=False)
        self.register_buffer("line_cell", line_cell, persistent=False)
        self.register_buffer("line_weights", weights, persistent=False)

        input_size = self.num_cells * 5
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
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
    ) -> PolicyValueLineMLP:
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

        hidden = self.trunk(self.input_layer(self._features(board)))
        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        if squeeze_batch:
            return policy_logits.squeeze(0), value.squeeze(0)
        return policy_logits, value

    def _features(self, board: torch.Tensor) -> torch.Tensor:
        own = (board > 0.5).to(dtype=torch.float32)
        opponent = (board < -0.5).to(dtype=torch.float32)
        line_own = own[:, self.lines].sum(dim=-1).to(dtype=torch.long)
        line_opponent = opponent[:, self.lines].sum(dim=-1).to(dtype=torch.long)
        open_own = torch.where(
            line_opponent == 0,
            self.line_weights[line_own],
            torch.zeros_like(line_own, dtype=torch.float32),
        )
        open_opponent = torch.where(
            line_own == 0,
            self.line_weights[line_opponent],
            torch.zeros_like(line_opponent, dtype=torch.float32),
        )
        own_cell_scores = open_own @ self.line_cell
        opponent_cell_scores = open_opponent @ self.line_cell
        return torch.cat(
            [
                board,
                own,
                opponent,
                torch.log1p(own_cell_scores),
                torch.log1p(opponent_cell_scores),
            ],
            dim=1,
        )

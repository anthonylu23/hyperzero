"""Small RL-style environment wrapper around GameState."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hyperzero.game.config import GameConfig
from hyperzero.game.state import GameState


@dataclass(slots=True)
class ConnectKEnv:
    """Turn-based environment wrapper for self-play and simple agents."""

    config: GameConfig
    use_line_counts: bool = False
    state: GameState = field(init=False)

    def __post_init__(self) -> None:
        self.state = GameState.new(self.config, use_line_counts=self.use_line_counts)

    def reset(self) -> dict[str, Any]:
        """Reset to the initial state and return the first observation."""
        self.state = GameState.new(self.config, use_line_counts=self.use_line_counts)
        return self.observation()

    def step(self, action: int) -> tuple[dict[str, Any], int, bool, dict[str, Any]]:
        """Apply action and return observation, reward, terminated, info."""
        move = self.state.make_move(action)
        reward = 0
        if self.state.terminal:
            if self.state.winner == move.player:
                reward = 1
            elif self.state.winner == -move.player:
                reward = -1

        info = {
            "move": move,
            "winner": self.state.winner,
            "ply": self.state.ply,
            "hash": int(self.state.zobrist_hash),
        }
        return self.observation(), reward, self.state.terminal, info

    def observation(self) -> dict[str, Any]:
        """Return the current agent-facing observation."""
        return {
            "board": self.state.canonical_board(),
            "legal_mask": self.state.legal_mask(),
            "player_to_move": self.state.player_to_move,
            "ply": self.state.ply,
            "hash": int(self.state.zobrist_hash),
        }

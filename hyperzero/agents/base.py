"""Common agent protocol."""

from __future__ import annotations

from typing import Protocol

from hyperzero.game.state import GameState


class Agent(Protocol):
    """Protocol implemented by all action-selecting agents."""

    name: str

    def select_action(self, state: GameState) -> int:
        """Return a legal action for the given state."""
        ...

    def reset(self) -> None:
        """Clear any per-game state."""
        ...

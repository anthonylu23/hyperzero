"""Uniform random baseline agent."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hyperzero.game.errors import TerminalStateError
from hyperzero.game.state import GameState


@dataclass(slots=True)
class RandomAgent:
    """Select uniformly among legal actions."""

    seed: int | None = None
    name: str = "random"
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        """Random agents do not keep per-game state."""

    def select_action(self, state: GameState) -> int:
        """Return a uniformly sampled legal action."""
        legal_actions = state.legal_actions()
        if legal_actions.size == 0:
            raise TerminalStateError("cannot select an action from a terminal state")
        return int(self.rng.choice(legal_actions))

"""Pure MCTS baseline agent."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hyperzero.game.errors import TerminalStateError
from hyperzero.game.state import GameState
from hyperzero.search.mcts import MCTSConfig, MCTSResult, run_mcts


@dataclass(slots=True)
class MCTSAgent:
    """Select actions using pure Monte Carlo Tree Search."""

    simulations: int = 100
    exploration: float = 1.41421356237
    rollout_limit: int | None = None
    seed: int | None = None
    name: str = "mcts"
    rng: np.random.Generator = field(init=False, repr=False)
    last_result: MCTSResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        """Clear search diagnostics from the previous move."""
        self.last_result = None

    def select_action(self, state: GameState) -> int:
        """Run MCTS and return the selected legal action."""
        if state.terminal or state.legal_actions().size == 0:
            raise TerminalStateError("cannot select an action from a terminal state")

        result = run_mcts(
            state,
            MCTSConfig(
                simulations=self.simulations,
                exploration=self.exploration,
                rollout_limit=self.rollout_limit,
            ),
            rng=self.rng,
        )
        self.last_result = result
        return result.action

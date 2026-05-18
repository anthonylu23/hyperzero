"""Minimal AlphaZero-style neural-guided search agent."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hyperzero.game.errors import TerminalStateError
from hyperzero.game.state import GameState
from hyperzero.search.puct import (
    PolicyValueEvaluator,
    PUCTConfig,
    PUCTResult,
    run_puct,
)


@dataclass(slots=True)
class AlphaZeroAgent:
    """Select actions using a policy-value evaluator and PUCT search."""

    evaluator: PolicyValueEvaluator
    simulations: int = 50
    c_puct: float = 1.5
    seed: int | None = None
    name: str = "alphazero"
    rng: np.random.Generator = field(init=False, repr=False)
    last_result: PUCTResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        """Clear search diagnostics from the previous move."""
        self.last_result = None

    def select_action(self, state: GameState) -> int:
        """Run neural-guided PUCT and return the selected legal action."""
        if state.terminal or state.legal_actions().size == 0:
            raise TerminalStateError("cannot select an action from a terminal state")

        result = run_puct(
            state,
            self.evaluator,
            PUCTConfig(simulations=self.simulations, c_puct=self.c_puct),
            rng=self.rng,
        )
        self.last_result = result
        return result.action

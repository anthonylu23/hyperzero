"""One-ply tactical baseline agent."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hyperzero.game.errors import TerminalStateError
from hyperzero.game.state import GameState


@dataclass(slots=True)
class TacticalAgent:
    """Prefer immediate wins, then moves that prevent immediate losses."""

    seed: int | None = None
    name: str = "tactical"
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        """Tactical agents do not keep per-game state."""

    def select_action(self, state: GameState) -> int:
        """Return a legal one-ply tactical action."""
        legal_actions = state.legal_actions()
        if legal_actions.size == 0:
            raise TerminalStateError("cannot select an action from a terminal state")

        player = state.player_to_move
        winning_actions = [
            int(action)
            for action in legal_actions
            if _action_outcome(state, int(action)) == player
        ]
        if winning_actions:
            return _sample_tie(winning_actions, self.rng)

        safe_actions = [
            int(action)
            for action in legal_actions
            if not _opponent_has_immediate_win_after(state, int(action))
        ]
        if safe_actions:
            return _sample_tie(safe_actions, self.rng)

        return int(self.rng.choice(legal_actions))


def _action_outcome(state: GameState, action: int) -> int | None:
    state.make_move(action)
    winner = state.winner if state.terminal else None
    state.undo_move()
    return winner


def _opponent_has_immediate_win_after(state: GameState, action: int) -> bool:
    player = state.player_to_move
    state.make_move(action)
    try:
        for response in state.legal_actions():
            if _action_outcome(state, int(response)) == -player:
                return True
        return False
    finally:
        state.undo_move()


def _sample_tie(actions: list[int], rng: np.random.Generator) -> int:
    return int(rng.choice(np.asarray(actions, dtype=np.int32)))

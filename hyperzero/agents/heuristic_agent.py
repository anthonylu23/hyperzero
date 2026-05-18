"""Line-scoring heuristic baseline agent."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hyperzero.game.errors import TerminalStateError
from hyperzero.game.state import GameState


@dataclass(slots=True)
class HeuristicAgent:
    """Score legal actions by open line potential and tactical threats."""

    seed: int | None = None
    own_scale: float = 1.0
    opponent_scale: float = 1.25
    center_scale: float = 0.05
    name: str = "heuristic"
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        """Heuristic agents do not keep per-game state."""

    def select_action(self, state: GameState) -> int:
        """Return the highest-scoring legal action."""
        legal_actions = state.legal_actions()
        if legal_actions.size == 0:
            raise TerminalStateError("cannot select an action from a terminal state")

        player = state.player_to_move
        best_score = -np.inf
        best_actions: list[int] = []
        for action in legal_actions:
            action = int(action)
            score = self.score_action(state, action, player)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return int(self.rng.choice(np.asarray(best_actions, dtype=np.int32)))

    def score_action(
        self,
        state: GameState,
        action: int,
        player: int | None = None,
    ) -> float:
        """Return a heuristic score for applying action in state."""
        player = state.player_to_move if player is None else player
        state.make_move(action)
        try:
            if state.terminal:
                if state.winner == player:
                    return np.inf
                if state.winner == -player:
                    return -np.inf
                return 0.0
            return self.score_state(state, player) + self._center_bonus(state, action)
        finally:
            state.undo_move()

    def score_state(self, state: GameState, player: int | None = None) -> float:
        """Return a line-based score from player's absolute perspective."""
        player = state.player_to_move if player is None else player
        opponent = -player
        score = 0.0

        for line in state.config.winning_lines:
            values = state.board[line]
            own_count = int(np.count_nonzero(values == player))
            opponent_count = int(np.count_nonzero(values == opponent))
            if own_count and opponent_count:
                continue
            if own_count:
                score += self.own_scale * _line_weight(
                    own_count,
                    state.config.connect_k,
                )
            elif opponent_count:
                score -= self.opponent_scale * _line_weight(
                    opponent_count,
                    state.config.connect_k,
                )

        return score

    def _center_bonus(self, state: GameState, action: int) -> float:
        if self.center_scale == 0.0 or not state.config.action_shape:
            return 0.0

        coord = np.asarray(state.config.column_coord(action), dtype=np.float64)
        center = (np.asarray(state.config.action_shape, dtype=np.float64) - 1.0) / 2.0
        max_distance = float(np.linalg.norm(center))
        if max_distance == 0.0:
            return self.center_scale
        distance = float(np.linalg.norm(coord - center))
        return self.center_scale * (1.0 - distance / max_distance)


def _line_weight(count: int, connect_k: int) -> float:
    if count >= connect_k:
        return 1_000_000.0
    return float(10 ** count)

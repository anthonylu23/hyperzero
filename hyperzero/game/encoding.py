"""Agent-facing state encodings."""

from __future__ import annotations

import numpy as np

from hyperzero.game.state import GameState


def canonical_board(state: GameState, *, flat: bool = False) -> np.ndarray:
    """Return the board from the current player's perspective."""
    board = state.board * state.player_to_move
    if flat:
        return board.copy()
    return board.reshape(state.config.shape).copy()


def legal_action_mask(state: GameState) -> np.ndarray:
    """Return a boolean mask over the full action space."""
    return state.legal_mask()

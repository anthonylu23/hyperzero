"""Mutable game state for N-dimensional Connect-K with gravity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from operator import index

import numpy as np

from hyperzero.game.config import GameConfig
from hyperzero.game.errors import InvalidActionError, TerminalStateError


@dataclass(frozen=True, slots=True)
class Move:
    """Resolved move information suitable for logs, replays, and GUI snapshots."""

    action: int
    column_coord: tuple[int, ...]
    cell_index: int
    cell_coord: tuple[int, ...]
    player: int
    ply: int


@dataclass(frozen=True, slots=True)
class _UndoRecord:
    action: int
    cell_index: int
    player: int
    previous_terminal: bool
    previous_winner: int | None
    previous_last_move: Move | None
    previous_hash: np.uint64


class GameState:
    """Mutable board state optimized for repeated simulation."""

    def __init__(self, config: GameConfig, *, use_line_counts: bool = False) -> None:
        self.config = config
        self.board = np.zeros(config.num_cells, dtype=np.int8)
        self.heights = np.zeros(config.num_actions, dtype=np.int16)
        self.use_line_counts = bool(use_line_counts)
        self.line_counts = (
            np.zeros((2, len(config.winning_lines)), dtype=np.int16)
            if self.use_line_counts
            else None
        )
        self.player_to_move = 1
        self.ply = 0
        self.terminal = False
        self.winner: int | None = None
        self.last_move: Move | None = None
        self.zobrist_hash = np.uint64(0)
        self._history: list[_UndoRecord] = []

    @classmethod
    def new(cls, config: GameConfig, *, use_line_counts: bool = False) -> GameState:
        """Create a fresh initial state."""
        return cls(config, use_line_counts=use_line_counts)

    def copy(self, *, include_history: bool = False) -> GameState:
        """Return a deep copy of this state."""
        copied = type(self)(self.config, use_line_counts=self.use_line_counts)
        copied.board = self.board.copy()
        copied.heights = self.heights.copy()
        if self.line_counts is not None:
            copied.line_counts = self.line_counts.copy()
        copied.player_to_move = self.player_to_move
        copied.ply = self.ply
        copied.terminal = self.terminal
        copied.winner = self.winner
        copied.last_move = self.last_move
        copied.zobrist_hash = np.uint64(self.zobrist_hash)
        if include_history:
            copied._history = list(self._history)
        return copied

    def apply(self, action: int) -> GameState:
        """Return a copied state after applying action."""
        next_state = self.copy(include_history=True)
        next_state.make_move(action)
        return next_state

    def make_move(self, action: int) -> Move:
        """Apply action in place and return the resolved move."""
        if self.terminal:
            raise TerminalStateError("cannot make a move from a terminal state")

        action = self._coerce_action(action)
        if self.heights[action] >= self.config.gravity_size:
            raise InvalidActionError(f"column for action {action} is full")

        previous_terminal = self.terminal
        previous_winner = self.winner
        previous_last_move = self.last_move
        previous_hash = np.uint64(self.zobrist_hash)
        player = self.player_to_move
        height = int(self.heights[action])
        cell_index = int(self.config.column_cells[action, height])

        self.board[cell_index] = player
        self.heights[action] += 1
        self.zobrist_hash ^= self.config.zobrist_piece(cell_index, player)

        move = Move(
            action=action,
            column_coord=self.config.column_coord(action),
            cell_index=cell_index,
            cell_coord=self.config.cell_coord(cell_index),
            player=player,
            ply=self.ply,
        )
        self._history.append(
            _UndoRecord(
                action=action,
                cell_index=cell_index,
                player=player,
                previous_terminal=previous_terminal,
                previous_winner=previous_winner,
                previous_last_move=previous_last_move,
                previous_hash=previous_hash,
            )
        )

        self.ply += 1
        self.last_move = move
        if self._is_winning_move(cell_index, player):
            self.terminal = True
            self.winner = player
        elif np.all(self.heights >= self.config.gravity_size):
            self.terminal = True
            self.winner = 0

        self.player_to_move = -self.player_to_move
        self.zobrist_hash ^= self.config.zobrist_side
        return move

    def undo_move(self) -> Move:
        """Undo the most recent move and return it."""
        if not self._history:
            raise InvalidActionError("cannot undo without move history")

        record = self._history.pop()
        undone = self.last_move
        if undone is None:
            raise RuntimeError("state history is inconsistent")

        self.board[record.cell_index] = 0
        self.heights[record.action] -= 1
        self._unmark_line_counts(record.cell_index, record.player)
        self.player_to_move = record.player
        self.ply -= 1
        self.terminal = record.previous_terminal
        self.winner = record.previous_winner
        self.last_move = record.previous_last_move
        self.zobrist_hash = np.uint64(record.previous_hash)

        return undone

    def legal_mask(self) -> np.ndarray:
        """Return a boolean mask over legal actions."""
        if self.terminal:
            return np.zeros(self.config.num_actions, dtype=bool)
        return self.heights < self.config.gravity_size

    def legal_actions(self) -> np.ndarray:
        """Return legal action ids."""
        return np.flatnonzero(self.legal_mask()).astype(np.int32)

    def canonical_board(self, *, flat: bool = False) -> np.ndarray:
        """Return the board from the current player's perspective."""
        board = self.board * self.player_to_move
        if flat:
            return board.copy()
        return board.reshape(self.config.shape).copy()

    def board_tensor(self) -> np.ndarray:
        """Return the absolute board tensor."""
        return self.board.reshape(self.config.shape).copy()

    def action_history(self) -> tuple[int, ...]:
        """Return the played actions in order."""
        return tuple(record.action for record in self._history)

    def recompute_hash(self) -> np.uint64:
        """Recompute and store the Zobrist hash from the current board."""
        zobrist_hash = np.uint64(0)
        for cell_index, value in enumerate(self.board):
            player = int(value)
            if player != 0:
                zobrist_hash ^= self.config.zobrist_piece(cell_index, player)
        if self.player_to_move == -1:
            zobrist_hash ^= self.config.zobrist_side
        self.zobrist_hash = np.uint64(zobrist_hash)
        return self.zobrist_hash

    def terminal_value(self) -> int:
        """Return terminal value from player_to_move's perspective."""
        if not self.terminal:
            raise ValueError("terminal_value is only defined for terminal states")
        if self.winner == 0:
            return 0
        return 1 if self.winner == self.player_to_move else -1

    def outcome_for_player(self, player: int) -> int:
        """Return the terminal outcome from an absolute player's perspective."""
        if player not in (-1, 1):
            raise ValueError("player must be 1 or -1")
        if not self.terminal:
            raise ValueError("outcome is only defined for terminal states")
        if self.winner == 0:
            return 0
        return 1 if self.winner == player else -1

    def to_snapshot(self) -> dict[str, object]:
        """Return a serializable view for replay, debugging, or GUI layers."""
        return {
            "shape": self.config.shape,
            "board": self.board_tensor().tolist(),
            "player_to_move": self.player_to_move,
            "ply": self.ply,
            "terminal": self.terminal,
            "winner": self.winner,
            "hash": int(self.zobrist_hash),
            "last_move": None if self.last_move is None else asdict(self.last_move),
            "legal_mask": self.legal_mask().tolist(),
        }

    def _coerce_action(self, action: int) -> int:
        try:
            action = index(action)
            self.config.validate_action_index(action)
        except (TypeError, ValueError) as exc:
            raise InvalidActionError(str(exc)) from exc
        return action

    def _is_winning_move(self, cell_index: int, player: int) -> bool:
        if self.line_counts is not None:
            return self._mark_line_counts(cell_index, player)

        for line_id in self.config.lines_by_cell[cell_index]:
            line = self.config.winning_lines[line_id]
            if np.all(self.board[line] == player):
                return True
        return False

    def _mark_line_counts(self, cell_index: int, player: int) -> bool:
        if self.line_counts is None:
            return False

        player_index = 1 if player == 1 else 0
        won = False
        for line_id in self.config.lines_by_cell[cell_index]:
            self.line_counts[player_index, line_id] += 1
            if self.line_counts[player_index, line_id] == self.config.connect_k:
                won = True
        return won

    def _unmark_line_counts(self, cell_index: int, player: int) -> None:
        if self.line_counts is None:
            return

        player_index = 1 if player == 1 else 0
        for line_id in self.config.lines_by_cell[cell_index]:
            self.line_counts[player_index, line_id] -= 1

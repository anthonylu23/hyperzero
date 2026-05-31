"""In-memory game sessions for the local web demo."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field

from hyperzero.game import GameState, InvalidActionError, TerminalStateError
from hyperzero.server.agent_service import AgentMoveResult, AgentService
from hyperzero.server.modes import ModeSpec, build_mode_spec, get_mode

DEFAULT_MODE_ID = "2d_6x7_k4"

PLAYER_BY_MARK = {"X": 1, "O": -1}
MARK_BY_PLAYER = {1: "X", -1: "O", 0: "Draw", None: None}


@dataclass(slots=True)
class GameSession:
    """One in-memory game between a human and the universal agent."""

    game_id: str
    mode: ModeSpec
    human_player: int
    difficulty: str
    state: GameState = field(init=False)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    agent_move_count: int = 0
    human_move_count: int = 0

    def __post_init__(self) -> None:
        self.state = GameState.new(self.mode.game_config)

    @property
    def agent_player(self) -> int:
        return -self.human_player

    def apply_human_move(self, action: int) -> dict[str, object]:
        """Apply a human move after checking turn ownership."""
        self._ensure_active()
        if self.state.player_to_move != self.human_player:
            raise TurnError("it is not the human player's turn")
        move = self.state.make_move(action)
        self.human_move_count += 1
        self.updated_at = time.time()
        return asdict(move)

    def apply_agent_move(
        self,
        agent_service: AgentService,
    ) -> tuple[dict[str, object], AgentMoveResult]:
        """Ask the universal agent for a move and apply it."""
        self._ensure_active()
        if self.state.player_to_move != self.agent_player:
            raise TurnError("it is not the agent player's turn")
        result = agent_service.select_action(
            self.state,
            difficulty=self.difficulty,
        )
        move = self.state.make_move(result.action)
        self.agent_move_count += 1
        self.updated_at = time.time()
        return asdict(move), result

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-safe game snapshot for clients."""
        state_snapshot = self.state.to_snapshot()
        config = self.state.config
        cells = [
            {
                "index": index,
                "coord": list(config.cell_coord(index)),
                "value": int(value),
            }
            for index, value in enumerate(self.state.board.tolist())
        ]
        legal_mask = self.state.legal_mask().tolist()
        actions = []
        for action in range(config.num_actions):
            height = int(self.state.heights[action])
            legal = bool(legal_mask[action])
            next_cell = (
                config.cell_coord(int(config.column_cells[action, height]))
                if legal
                else None
            )
            actions.append(
                {
                    "action": action,
                    "coord": list(config.column_coord(action)),
                    "legal": legal,
                    "height": height,
                    "next_cell": None if next_cell is None else list(next_cell),
                }
            )

        winner = self.state.winner
        return {
            "game_id": self.game_id,
            "mode": self.mode.to_dict(),
            "human_player": self.human_player,
            "human_mark": MARK_BY_PLAYER[self.human_player],
            "agent_player": self.agent_player,
            "agent_mark": MARK_BY_PLAYER[self.agent_player],
            "difficulty": self.difficulty,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turn": self.state.player_to_move,
            "turn_mark": MARK_BY_PLAYER[self.state.player_to_move],
            "winner": winner,
            "winner_mark": MARK_BY_PLAYER[winner],
            "terminal": self.state.terminal,
            "is_human_turn": (
                not self.state.terminal
                and self.state.player_to_move == self.human_player
            ),
            "is_agent_turn": (
                not self.state.terminal
                and self.state.player_to_move == self.agent_player
            ),
            "ply": self.state.ply,
            "human_move_count": self.human_move_count,
            "agent_move_count": self.agent_move_count,
            "state": state_snapshot,
            "cells": cells,
            "actions": actions,
            "winning_cells": self._winning_cells(),
        }

    def _winning_cells(self) -> list[int]:
        """Return the flat cell indices of the winning line, or [] if none.

        The last move is always the winning move on a decisive terminal state,
        so the winning line is the one through that cell fully owned by winner.
        """
        state = self.state
        if not state.terminal or state.winner in (None, 0):
            return []
        last_move = state.last_move
        if last_move is None:
            return []
        config = state.config
        winner = state.winner
        board = state.board
        for line_id in config.lines_by_cell[last_move.cell_index]:
            line = config.winning_lines[line_id]
            if all(int(board[cell]) == winner for cell in line):
                return [int(cell) for cell in line]
        return []

    def _ensure_active(self) -> None:
        if self.state.terminal:
            raise TerminalStateError("game is already finished")


class TurnError(RuntimeError):
    """Raised when a player tries to act out of turn."""


class SessionManager:
    """Store local demo games in memory."""

    def __init__(self, agent_service: AgentService | None = None) -> None:
        self.agent_service = AgentService() if agent_service is None else agent_service
        self._sessions: dict[str, GameSession] = {}

    def create_game(
        self,
        *,
        mode_id: str | None = None,
        shape: tuple[int, ...] | list[int] | None = None,
        connect_k: int | None = None,
        gravity_axis: int = 0,
        human_mark: str,
        difficulty: str,
    ) -> GameSession:
        if shape is not None:
            if connect_k is None:
                raise ValueError("connect_k is required when shape is provided")
            mode = build_mode_spec(tuple(shape), connect_k, gravity_axis)
        else:
            mode = get_mode(mode_id or DEFAULT_MODE_ID)
        human_player = player_from_mark(human_mark)
        game = GameSession(
            game_id=uuid.uuid4().hex,
            mode=mode,
            human_player=human_player,
            difficulty=difficulty,
        )
        self._sessions[game.game_id] = game
        return game

    def get_game(self, game_id: str) -> GameSession:
        try:
            return self._sessions[game_id]
        except KeyError as exc:
            raise KeyError(f"unknown game id: {game_id}") from exc

    def apply_human_move(
        self,
        game_id: str,
        action: int,
    ) -> tuple[GameSession, dict[str, object]]:
        game = self.get_game(game_id)
        move = game.apply_human_move(action)
        return game, move

    def apply_agent_move(
        self,
        game_id: str,
    ) -> tuple[GameSession, dict[str, object], AgentMoveResult]:
        game = self.get_game(game_id)
        move, result = game.apply_agent_move(self.agent_service)
        return game, move, result


def player_from_mark(mark: str) -> int:
    """Return the internal player value for an X/O mark."""
    normalized = mark.upper()
    try:
        return PLAYER_BY_MARK[normalized]
    except KeyError as exc:
        raise ValueError("player mark must be X or O") from exc


def error_name(exc: Exception) -> str:
    """Return a stable public error category for API responses."""
    if isinstance(exc, InvalidActionError):
        return "invalid_action"
    if isinstance(exc, TerminalStateError):
        return "terminal_state"
    if isinstance(exc, TurnError):
        return "wrong_turn"
    if isinstance(exc, KeyError):
        return "not_found"
    if isinstance(exc, ValueError):
        return "invalid_request"
    return "server_error"

"""FastAPI entrypoint for the local HyperZero playable demo."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hyperzero.game import InvalidActionError, TerminalStateError
from hyperzero.server.agent_service import AgentMoveResult, simulations_for_difficulty
from hyperzero.server.modes import DEMO_MODES
from hyperzero.server.sessions import SessionManager, TurnError, error_name


class CreateGameRequest(BaseModel):
    mode_id: str | None = Field(default="2d_6x7_k4")
    shape: list[int] | None = None
    connect_k: int | None = None
    gravity_axis: int = 0
    human_mark: Literal["X", "O"] = "X"
    difficulty: Literal["quick", "normal", "strong"] = "normal"


class MoveRequest(BaseModel):
    action: int = Field(ge=0)


app = FastAPI(title="HyperZero Local Demo API")
LOCAL_DEV_ORIGIN_REGEX = r"^http://(localhost|127\.0\.0\.1|\[::1\]):\d+$"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=LOCAL_DEV_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = SessionManager()


@app.get("/health")
def health() -> dict[str, object]:
    """Return API and model availability."""
    return {
        "ok": True,
        "model": manager.agent_service.metadata(),
    }


@app.get("/modes")
def modes() -> dict[str, object]:
    """Return supported local demo modes."""
    return {
        "modes": [mode.to_dict() for mode in DEMO_MODES.values()],
        "difficulties": {
            difficulty: {"simulations": simulations_for_difficulty(difficulty)}
            for difficulty in ("quick", "normal", "strong")
        },
    }


@app.get("/model")
def model() -> dict[str, object]:
    """Return universal checkpoint metadata."""
    return manager.agent_service.metadata()


@app.post("/games")
def create_game(request: CreateGameRequest) -> dict[str, object]:
    """Create a new in-memory game."""
    try:
        game = manager.create_game(
            mode_id=request.mode_id,
            shape=request.shape,
            connect_k=request.connect_k,
            gravity_axis=request.gravity_axis,
            human_mark=request.human_mark,
            difficulty=request.difficulty,
        )
    except ValueError as exc:
        raise _http_error(exc) from exc
    return {"game": game.snapshot()}


@app.get("/games/{game_id}")
def get_game(game_id: str) -> dict[str, object]:
    """Return the current game snapshot."""
    try:
        game = manager.get_game(game_id)
    except KeyError as exc:
        raise _http_error(exc) from exc
    return {"game": game.snapshot()}


@app.post("/games/{game_id}/moves")
def apply_move(game_id: str, request: MoveRequest) -> dict[str, object]:
    """Apply a human move."""
    try:
        game, move = manager.apply_human_move(game_id, request.action)
    except (
        InvalidActionError,
        KeyError,
        TerminalStateError,
        TurnError,
        ValueError,
    ) as exc:
        raise _http_error(exc) from exc
    return {
        "move": move,
        "game": game.snapshot(),
    }


@app.post("/games/{game_id}/agent-move")
def apply_agent_move(game_id: str) -> dict[str, object]:
    """Compute and apply a universal-agent move."""
    try:
        game, move, result = manager.apply_agent_move(game_id)
    except (
        FileNotFoundError,
        InvalidActionError,
        KeyError,
        TerminalStateError,
        TurnError,
        ValueError,
    ) as exc:
        raise _http_error(exc) from exc
    return {
        "move": move,
        "agent": _agent_result_payload(result),
        "game": game.snapshot(),
    }


def _agent_result_payload(result: AgentMoveResult) -> dict[str, object]:
    return {
        "action": result.action,
        "duration_ms": result.duration_ms,
        "simulations": result.simulations,
        "value": result.value,
        "visits": list(result.visits),
        "policy": list(result.policy),
    }


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        status_code = 404
    elif isinstance(exc, (TerminalStateError, TurnError)):
        status_code = 409
    elif isinstance(exc, FileNotFoundError):
        status_code = 503
    elif isinstance(exc, (InvalidActionError, ValueError)):
        status_code = 400
    else:
        status_code = 500
    return HTTPException(
        status_code=status_code,
        detail={
            "error": error_name(exc),
            "message": str(exc),
        },
    )

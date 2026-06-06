"""Universal checkpoint inference service for the local API."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hyperzero.game import GameState
from hyperzero.game.actions import normalize_legal_policy
from hyperzero.models import UniversalEvaluator
from hyperzero.search.puct import PUCTConfig, PUCTResult, PUCTSearchSession, run_puct
from hyperzero.training import (
    LoadedUniversalCheckpoint,
    load_universal_training_checkpoint,
)

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "universal_residual_followup_20260528"
    / "residual_recovery_teacher010_lr2e5_seed6604"
    / "checkpoints"
    / "best_by_eval_score.pt"
)

DIFFICULTY_SIMULATIONS = {
    "quick": 4,
    "normal": 8,
    "strong": 16,
}


@dataclass(frozen=True, slots=True)
class AgentMoveResult:
    """Model move result plus search diagnostics."""

    action: int
    duration_ms: float
    simulations: int
    value: float
    visits: tuple[int, ...]
    policy: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AgentSearchEvent:
    """One server-streamed update from agent search."""

    event: str
    payload: dict[str, object]


class AgentService:
    """Lazy loader for the universal model and PUCT move selection."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        env_checkpoint = os.environ.get("HYPERZERO_UNIVERSAL_CHECKPOINT")
        self.checkpoint_path = Path(
            checkpoint_path or env_checkpoint or DEFAULT_CHECKPOINT,
        )
        self.device = device or os.environ.get("HYPERZERO_DEVICE", "cpu")
        self.seed = seed
        self._checkpoint: LoadedUniversalCheckpoint | None = None
        self._evaluator: UniversalEvaluator | None = None
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._checkpoint is not None

    @property
    def checkpoint(self) -> LoadedUniversalCheckpoint:
        return self._load()

    def preload(self) -> None:
        """Load the checkpoint and evaluator before serving traffic."""
        self._load_evaluator()

    def metadata(self) -> dict[str, object]:
        """Return model metadata without forcing a model load when possible."""
        loaded = self._checkpoint
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "device": self.device,
            "loaded": loaded is not None,
            "iteration": None if loaded is None else loaded.iteration,
            "game_specs": []
            if loaded is None
            else [spec.to_dict() for spec in loaded.game_specs],
            "difficulties": {
                name: {"simulations": simulations}
                for name, simulations in DIFFICULTY_SIMULATIONS.items()
            },
        }

    def select_action(
        self,
        state: GameState,
        *,
        difficulty: str,
    ) -> AgentMoveResult:
        """Run PUCT from the given state and return the selected move."""
        simulations = simulations_for_difficulty(difficulty)
        with self._lock:
            evaluator = self._load_evaluator()
            start = time.perf_counter()
            result = run_puct(
                state,
                evaluator,
                PUCTConfig(simulations=simulations, c_puct=1.5),
                rng=self._rng,
            )
            if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
                torch.cuda.synchronize(torch.device(self.device))
            duration_ms = (time.perf_counter() - start) * 1000.0
        return _to_agent_move_result(result, duration_ms, simulations)

    def select_action_events(
        self,
        state: GameState,
        *,
        difficulty: str,
    ) -> Iterator[AgentSearchEvent]:
        """Run PUCT and yield progress events ending with a final move result."""
        simulations = simulations_for_difficulty(difficulty)
        yield AgentSearchEvent(
            "model_loading",
            {
                "difficulty": difficulty,
                "simulations": simulations,
                "loaded": self.loaded,
            },
        )
        with self._lock:
            evaluator = self._load_evaluator()
            config = PUCTConfig(simulations=simulations, c_puct=1.5)
            session = PUCTSearchSession(state, config, self._rng)
            start = time.perf_counter()

            leaf = session.select_leaf()
            session.complete_leaf(leaf, evaluator.evaluate(leaf.state))
            yield AgentSearchEvent(
                "search_started",
                _search_progress_payload(session, state, start, simulations),
            )

            while not session.complete:
                leaf = session.select_leaf()
                evaluation = (
                    evaluator.evaluate(leaf.state) if leaf.requires_evaluation else None
                )
                session.complete_leaf(leaf, evaluation)
                yield AgentSearchEvent(
                    "simulation_progress",
                    _search_progress_payload(session, state, start, simulations),
                )

            result = session.result()
            if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
                torch.cuda.synchronize(torch.device(self.device))
            duration_ms = (time.perf_counter() - start) * 1000.0

        yield AgentSearchEvent(
            "move_final",
            {
                "agent": _agent_result_payload(
                    _to_agent_move_result(result, duration_ms, simulations)
                ),
            },
        )

    def _load(self) -> LoadedUniversalCheckpoint:
        if self._checkpoint is None:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(
                    f"universal checkpoint does not exist: {self.checkpoint_path}"
                )
            self._checkpoint = load_universal_training_checkpoint(
                self.checkpoint_path,
                device=self.device,
            )
        return self._checkpoint

    def _load_evaluator(self) -> UniversalEvaluator:
        if self._evaluator is None:
            checkpoint = self._load()
            self._evaluator = UniversalEvaluator(
                checkpoint.model,
                checkpoint.model_config.encoder,
                device=self.device,
            )
        return self._evaluator


def simulations_for_difficulty(difficulty: str) -> int:
    """Return the fixed PUCT budget for a user-facing difficulty."""
    try:
        return DIFFICULTY_SIMULATIONS[difficulty]
    except KeyError as exc:
        known = ", ".join(sorted(DIFFICULTY_SIMULATIONS))
        raise ValueError(
            f"unknown difficulty {difficulty!r}; expected one of {known}"
        ) from exc


def _to_agent_move_result(
    result: PUCTResult,
    duration_ms: float,
    simulations: int,
) -> AgentMoveResult:
    return AgentMoveResult(
        action=int(result.action),
        duration_ms=duration_ms,
        simulations=simulations,
        value=float(result.root.mean_value),
        visits=tuple(int(value) for value in result.visits.tolist()),
        policy=tuple(float(value) for value in result.policy.tolist()),
    )


def _agent_result_payload(result: AgentMoveResult) -> dict[str, object]:
    return {
        "action": result.action,
        "duration_ms": result.duration_ms,
        "simulations": result.simulations,
        "value": result.value,
        "visits": list(result.visits),
        "policy": list(result.policy),
    }


def _search_progress_payload(
    session: PUCTSearchSession,
    state: GameState,
    start: float,
    simulations: int,
) -> dict[str, object]:
    visits = np.zeros(state.config.num_actions, dtype=np.float64)
    for action, child in session.root.children.items():
        visits[action] = child.visit_count
    policy = normalize_legal_policy(visits, state.legal_mask())
    return {
        "simulations_completed": session.simulations_run,
        "simulations": simulations,
        "duration_ms": (time.perf_counter() - start) * 1000.0,
        "visits": [int(value) for value in visits.tolist()],
        "policy": [float(value) for value in policy.tolist()],
    }

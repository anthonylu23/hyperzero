"""Universal checkpoint inference service for the local API."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hyperzero.game import GameState
from hyperzero.models import UniversalEvaluator
from hyperzero.search.puct import PUCTConfig, PUCTResult, run_puct
from hyperzero.training import (
    LoadedUniversalCheckpoint,
    load_universal_training_checkpoint,
)

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "universal_4d_line_distill_bounded_20260523"
    / "distill_iter24_bounded_s800.pt"
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

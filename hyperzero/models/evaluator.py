"""Single-state neural policy-value evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from hyperzero.game.state import GameState
from hyperzero.search.puct import PolicyValueEvaluation


class PolicyValueModel(Protocol):
    """Torch module interface needed by the evaluator."""

    def eval(self) -> PolicyValueModel:
        """Set inference mode and return self."""
        ...

    def __call__(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits and value predictions."""
        ...


@dataclass(slots=True)
class NeuralEvaluator:
    """Evaluate game states with a policy-value model."""

    model: PolicyValueModel
    device: str | torch.device = "cpu"
    inference_time_seconds: float = 0.0
    inference_batches: int = 0
    inference_states: int = 0

    def evaluate(self, state: GameState) -> PolicyValueEvaluation:
        """Return raw model outputs for one state."""
        return self.evaluate_many([state])[0]

    def evaluate_many(self, states: list[GameState]) -> list[PolicyValueEvaluation]:
        """Return raw model outputs for multiple states in one model call."""
        if not states:
            raise ValueError("states must be nonempty")
        device = torch.device(self.device)
        board = torch.as_tensor(
            np.stack([state.canonical_board(flat=True) for state in states]),
            dtype=torch.float32,
            device=device,
        )

        self.model.eval()
        with torch.no_grad():
            self._synchronize_if_cuda(device)
            start = time.perf_counter()
            policy_logits, value = self.model(board)
            self._synchronize_if_cuda(device)
            self.inference_time_seconds += time.perf_counter() - start
            self.inference_batches += 1
            self.inference_states += len(states)

        logits_array = policy_logits.detach().cpu().numpy().astype(np.float64)
        value_array = value.detach().cpu().numpy().astype(np.float64)
        return [
            PolicyValueEvaluation(
                policy_logits=logits_array[index],
                value=max(-1.0, min(1.0, float(value_array[index]))),
            )
            for index in range(len(states))
        ]

    @staticmethod
    def _synchronize_if_cuda(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

"""Evaluator adapter for universal policy-value models."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from hyperzero.game.state import GameState
from hyperzero.search.puct import PolicyValueEvaluation
from hyperzero.universal.encoding import (
    UniversalEncoderConfig,
    collate_positions,
    encode_state,
)


@dataclass(slots=True)
class UniversalEvaluator:
    """Evaluate arbitrary supported GameState variants with one model."""

    model: torch.nn.Module
    encoder_config: UniversalEncoderConfig = UniversalEncoderConfig()
    device: str | torch.device = "cpu"
    inference_time_seconds: float = 0.0
    inference_batches: int = 0
    inference_states: int = 0

    def evaluate(self, state: GameState) -> PolicyValueEvaluation:
        """Return raw model outputs for one state."""
        return self.evaluate_many([state])[0]

    def evaluate_many(self, states: list[GameState]) -> list[PolicyValueEvaluation]:
        """Return per-state, unpadded policy logits and values."""
        if not states:
            raise ValueError("states must be nonempty")
        device = torch.device(self.device)
        positions = [encode_state(state, self.encoder_config) for state in states]
        batch = collate_positions(positions, device=device)

        self.model.eval()
        with torch.inference_mode():
            self._synchronize_if_cuda(device)
            start = time.perf_counter()
            policy_logits, value = self.model(batch)
            self._synchronize_if_cuda(device)
            self.inference_time_seconds += time.perf_counter() - start
            self.inference_batches += 1
            self.inference_states += len(states)

        logits_array = policy_logits.detach().cpu().numpy().astype(np.float64)
        value_array = value.detach().cpu().numpy().astype(np.float64)
        return [
            PolicyValueEvaluation(
                policy_logits=logits_array[index, : state.config.num_actions],
                value=max(-1.0, min(1.0, float(value_array[index]))),
            )
            for index, state in enumerate(states)
        ]

    @staticmethod
    def _synchronize_if_cuda(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

"""Replay storage for mixed-variant universal training."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hyperzero.game.config import GameConfig


@dataclass(frozen=True, slots=True)
class UniversalSelfPlayExample:
    """One policy/value target with its originating game configuration."""

    config_id: str
    game_config: GameConfig
    board: np.ndarray
    policy: np.ndarray
    value: float
    legal_mask: np.ndarray
    player_to_move: int
    ply: int


@dataclass(slots=True)
class UniversalReplayBuffer:
    """Fixed-capacity replay buffer with balanced mixed-config sampling."""

    capacity: int
    seed: int | None = None
    examples: deque[UniversalSelfPlayExample] = field(init=False, repr=False)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.examples = deque(maxlen=int(self.capacity))
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return len(self.examples)

    def add(self, example: UniversalSelfPlayExample) -> None:
        """Append one example, evicting oldest examples when full."""
        self.examples.append(example)

    def add_many(self, examples: list[UniversalSelfPlayExample]) -> None:
        """Append many examples in order."""
        for example in examples:
            self.add(example)

    def sample(
        self,
        batch_size: int,
        *,
        balanced: bool = True,
    ) -> list[UniversalSelfPlayExample]:
        """Return a minibatch, optionally balancing across config ids."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.examples:
            raise ValueError("cannot sample from an empty replay buffer")
        if not balanced:
            return self._sample_flat(batch_size)
        return self._sample_balanced(batch_size)

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable replay state for checkpoints."""
        return {
            "capacity": self.capacity,
            "examples": list(self.examples),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore replay examples and RNG state."""
        capacity = int(state["capacity"])
        if capacity != self.capacity:
            raise ValueError(
                f"replay capacity mismatch: checkpoint={capacity} "
                f"config={self.capacity}"
            )
        self.examples = deque(state["examples"], maxlen=self.capacity)
        self.rng.bit_generator.state = state["rng_state"]

    def counts_by_config(self) -> dict[str, int]:
        """Return current replay population by variant id."""
        counts: dict[str, int] = defaultdict(int)
        for example in self.examples:
            counts[example.config_id] += 1
        return dict(counts)

    def _sample_flat(self, batch_size: int) -> list[UniversalSelfPlayExample]:
        size = min(int(batch_size), len(self.examples))
        indices = self.rng.choice(len(self.examples), size=size, replace=False)
        snapshot = list(self.examples)
        return [snapshot[int(index)] for index in indices]

    def _sample_balanced(self, batch_size: int) -> list[UniversalSelfPlayExample]:
        by_config: dict[str, list[UniversalSelfPlayExample]] = defaultdict(list)
        for example in self.examples:
            by_config[example.config_id].append(example)
        config_ids = sorted(by_config)
        if not config_ids:
            return []

        target_size = min(int(batch_size), len(self.examples))
        selected: list[UniversalSelfPlayExample] = []
        cursor = 0
        while len(selected) < target_size:
            config_id = config_ids[cursor % len(config_ids)]
            bucket = by_config[config_id]
            selected.append(bucket[int(self.rng.integers(len(bucket)))])
            cursor += 1
        self.rng.shuffle(selected)
        return selected

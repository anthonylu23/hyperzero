"""Simple in-memory replay buffer for v1 training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from hyperzero.training.self_play import SelfPlayExample


@dataclass(slots=True)
class ReplayBuffer:
    """Fixed-capacity replay buffer with random minibatch sampling."""

    capacity: int
    seed: int | None = None
    examples: deque[SelfPlayExample] = field(init=False, repr=False)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.examples = deque(maxlen=int(self.capacity))
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return len(self.examples)

    def add(self, example: SelfPlayExample) -> None:
        """Append one example, evicting the oldest if full."""
        self.examples.append(example)

    def add_many(
        self,
        examples: tuple[SelfPlayExample, ...] | list[SelfPlayExample],
    ) -> None:
        """Append many examples in order."""
        for example in examples:
            self.add(example)

    def sample(self, batch_size: int) -> list[SelfPlayExample]:
        """Return a random minibatch without replacement."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.examples:
            raise ValueError("cannot sample from an empty replay buffer")

        size = min(int(batch_size), len(self.examples))
        indices = self.rng.choice(len(self.examples), size=size, replace=False)
        snapshot = list(self.examples)
        return [snapshot[int(index)] for index in indices]

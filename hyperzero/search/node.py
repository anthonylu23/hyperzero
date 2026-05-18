"""Tree nodes for Monte Carlo Tree Search."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class MCTSNode:
    """A state node storing values from its player-to-move perspective."""

    player_to_move: int
    action: int | None = None
    parent: MCTSNode | None = None
    untried_actions: list[int] = field(default_factory=list)
    children: dict[int, MCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        """Return average value from this node's player-to-move perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_fully_expanded(self) -> bool:
        """Return whether all legal actions have child nodes."""
        return not self.untried_actions

    def best_child(self, exploration: float, rng: np.random.Generator) -> MCTSNode:
        """Return child maximizing UCT from this node's perspective."""
        if not self.children:
            raise ValueError("cannot select a child from a leaf node")

        log_parent = np.log(max(self.visit_count, 1))
        best_score = -np.inf
        best_children: list[MCTSNode] = []
        for child in self.children.values():
            if child.visit_count == 0:
                score = np.inf
            else:
                exploitation = -child.mean_value
                exploration_bonus = exploration * np.sqrt(
                    log_parent / child.visit_count,
                )
                score = exploitation + exploration_bonus

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return best_children[int(rng.integers(len(best_children)))]

"""Neural-guided PUCT search for AlphaZero-style agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from hyperzero.game.actions import logits_to_policy, normalize_legal_policy
from hyperzero.game.state import GameState


@dataclass(frozen=True, slots=True)
class PolicyValueEvaluation:
    """Raw policy logits and value from the current player's perspective."""

    policy_logits: np.ndarray
    value: float


class PolicyValueEvaluator(Protocol):
    """Evaluator interface consumed by PUCT search."""

    def evaluate(self, state: GameState) -> PolicyValueEvaluation:
        """Return policy logits and value for state."""
        ...

    def evaluate_many(self, states: list[GameState]) -> list[PolicyValueEvaluation]:
        """Return policy logits and value for multiple states."""
        ...


@dataclass(slots=True)
class PUCTNode:
    """A PUCT node storing value from its player-to-move perspective."""

    player_to_move: int
    prior: float = 0.0
    action: int | None = None
    parent: PUCTNode | None = None
    children: dict[int, PUCTNode] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    expanded: bool = False

    @property
    def mean_value(self) -> float:
        """Return average value from this node's player-to-move perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def best_child(self, c_puct: float, rng: np.random.Generator) -> PUCTNode:
        """Return the child maximizing PUCT from this node's perspective."""
        if not self.children:
            raise ValueError("cannot select a child from a leaf node")

        parent_visits = max(self.visit_count, 1)
        best_score = -np.inf
        best_children: list[PUCTNode] = []
        for child in self.children.values():
            exploitation = -child.mean_value
            exploration = (
                c_puct
                * child.prior
                * np.sqrt(parent_visits)
                / (1 + child.visit_count)
            )
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return best_children[int(rng.integers(len(best_children)))]


@dataclass(frozen=True, slots=True)
class PUCTConfig:
    """Configuration for neural-guided PUCT search."""

    simulations: int = 50
    c_puct: float = 1.5
    root_tactical_guard: bool = True

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("simulations must be positive")
        if self.c_puct < 0.0:
            raise ValueError("c_puct must be nonnegative")


@dataclass(frozen=True, slots=True)
class PUCTResult:
    """Search output for one root state."""

    action: int
    policy: np.ndarray
    visits: np.ndarray
    root: PUCTNode


@dataclass(frozen=True, slots=True)
class PUCTLeaf:
    """One selected leaf awaiting evaluation or terminal backup."""

    state: GameState
    node: PUCTNode
    path: tuple[PUCTNode, ...]
    terminal_value: float | None = None

    @property
    def requires_evaluation(self) -> bool:
        """Return whether this leaf needs a neural evaluation."""
        return self.terminal_value is None


@dataclass(slots=True)
class PUCTSearchSession:
    """Stepwise PUCT search state for batched leaf evaluation."""

    state: GameState
    config: PUCTConfig
    rng: np.random.Generator
    root: PUCTNode = field(init=False)
    simulations_run: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.state.terminal:
            raise ValueError("cannot run PUCT from a terminal state")
        self.root = PUCTNode(player_to_move=self.state.player_to_move)

    @property
    def complete(self) -> bool:
        """Return whether this session has run its configured simulations."""
        return self.simulations_run >= self.config.simulations

    @property
    def initialized(self) -> bool:
        """Return whether the root has been expanded."""
        return self.root.expanded

    def select_leaf(self) -> PUCTLeaf:
        """Select one leaf for evaluation or terminal backup."""
        if self.complete:
            raise ValueError("cannot select a leaf from a complete search")
        if not self.initialized:
            return PUCTLeaf(
                state=self.state.copy(include_history=True),
                node=self.root,
                path=(self.root,),
            )

        simulation_state = self.state.copy(include_history=True)
        node = self.root
        path = [node]

        while not simulation_state.terminal and node.expanded:
            node = node.best_child(self.config.c_puct, self.rng)
            if node.action is None:
                raise RuntimeError("non-root PUCT child is missing an action")
            simulation_state.make_move(node.action)
            path.append(node)

        if simulation_state.terminal:
            return PUCTLeaf(
                state=simulation_state,
                node=node,
                path=tuple(path),
                terminal_value=float(simulation_state.terminal_value()),
            )
        return PUCTLeaf(state=simulation_state, node=node, path=tuple(path))

    def complete_leaf(
        self,
        leaf: PUCTLeaf,
        evaluation: PolicyValueEvaluation | None = None,
    ) -> None:
        """Expand/evaluate a selected leaf and back up its value."""
        was_initialized = self.initialized
        if leaf.terminal_value is None:
            if evaluation is None:
                raise ValueError("nonterminal PUCT leaf requires an evaluation")
            value = _expand_with_evaluation(leaf.node, leaf.state, evaluation)
        else:
            value = leaf.terminal_value
        if not was_initialized:
            return
        _backup(list(leaf.path), value, leaf.state.player_to_move)
        self.simulations_run += 1

    def result(self) -> PUCTResult:
        """Return the root action and visit policy."""
        if not self.initialized:
            raise ValueError("cannot build a result before root expansion")
        visits = np.zeros(self.state.config.num_actions, dtype=np.float64)
        values = np.full(self.state.config.num_actions, -np.inf, dtype=np.float64)
        for action, child in self.root.children.items():
            visits[action] = child.visit_count
            values[action] = -child.mean_value

        legal_mask = self.state.legal_mask()
        policy = normalize_legal_policy(visits, legal_mask)
        if self.config.root_tactical_guard:
            policy = _apply_root_tactical_guard(self.state, policy)
        action = _select_root_action(policy, values, legal_mask, self.rng)
        return PUCTResult(action=action, policy=policy, visits=visits, root=self.root)


def run_puct(
    state: GameState,
    evaluator: PolicyValueEvaluator,
    config: PUCTConfig | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> PUCTResult:
    """Run non-batched neural-guided PUCT from state."""
    config = PUCTConfig() if config is None else config
    rng = np.random.default_rng() if rng is None else rng
    if state.terminal:
        raise ValueError("cannot run PUCT from a terminal state")

    session = PUCTSearchSession(state, config, rng)
    leaf = session.select_leaf()
    session.complete_leaf(leaf, evaluator.evaluate(leaf.state))
    while not session.complete:
        leaf = session.select_leaf()
        evaluation = (
            evaluator.evaluate(leaf.state) if leaf.requires_evaluation else None
        )
        session.complete_leaf(leaf, evaluation)
    return session.result()


def _evaluate_and_expand(
    node: PUCTNode,
    state: GameState,
    evaluator: PolicyValueEvaluator,
) -> float:
    return _expand_with_evaluation(node, state, evaluator.evaluate(state))


def _expand_with_evaluation(
    node: PUCTNode,
    state: GameState,
    evaluation: PolicyValueEvaluation,
) -> float:
    legal_mask = state.legal_mask()
    priors = logits_to_policy(evaluation.policy_logits, legal_mask)
    for action in np.flatnonzero(legal_mask):
        action = int(action)
        node.children[action] = PUCTNode(
            player_to_move=-state.player_to_move,
            prior=float(priors[action]),
            action=action,
            parent=node,
        )
    node.expanded = True
    return float(evaluation.value)


def _backup(path: list[PUCTNode], value: float, value_player: int) -> None:
    for node in path:
        node.visit_count += 1
        if node.player_to_move == value_player:
            node.value_sum += value
        else:
            node.value_sum -= value


def _apply_root_tactical_guard(state: GameState, policy: np.ndarray) -> np.ndarray:
    """Force root policy to take wins and avoid one-ply losses when possible."""
    legal_mask = state.legal_mask()
    winning_mask = np.zeros_like(legal_mask, dtype=bool)
    safe_mask = np.zeros_like(legal_mask, dtype=bool)
    for action in np.flatnonzero(legal_mask):
        action = int(action)
        if _action_wins(state, action, state.player_to_move):
            winning_mask[action] = True
        elif not _opponent_has_immediate_win_after(state, action):
            safe_mask[action] = True

    if winning_mask.any():
        return normalize_legal_policy(winning_mask.astype(np.float64), winning_mask)
    if not safe_mask.any():
        return policy

    guarded = np.where(safe_mask, policy, 0.0)
    if guarded.sum() > 0.0:
        return guarded / guarded.sum()
    return normalize_legal_policy(safe_mask.astype(np.float64), safe_mask)


def _action_wins(state: GameState, action: int, player: int) -> bool:
    state.make_move(action)
    try:
        return state.terminal and state.winner == player
    finally:
        state.undo_move()


def _opponent_has_immediate_win_after(state: GameState, action: int) -> bool:
    player = state.player_to_move
    state.make_move(action)
    try:
        for response in state.legal_actions():
            if _action_wins(state, int(response), -player):
                return True
        return False
    finally:
        state.undo_move()


def _select_root_action(
    policy: np.ndarray,
    values: np.ndarray,
    legal_mask: np.ndarray,
    rng: np.random.Generator,
) -> int:
    legal_actions = np.flatnonzero(legal_mask)
    legal_policy = policy[legal_actions]
    max_probability = legal_policy.max()
    candidates = legal_actions[legal_policy == max_probability]
    if candidates.size == 1:
        return int(candidates[0])

    candidate_values = values[candidates]
    max_value = candidate_values.max()
    candidates = candidates[candidate_values == max_value]
    return int(rng.choice(candidates))

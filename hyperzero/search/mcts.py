"""Pure Monte Carlo Tree Search for Connect-K."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hyperzero.game.actions import normalize_legal_policy
from hyperzero.game.state import GameState
from hyperzero.search.node import MCTSNode


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """Configuration for pure Monte Carlo Tree Search."""

    simulations: int = 100
    exploration: float = 1.41421356237
    rollout_limit: int | None = None

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("simulations must be positive")
        if self.exploration < 0.0:
            raise ValueError("exploration must be nonnegative")
        if self.rollout_limit is not None and self.rollout_limit <= 0:
            raise ValueError("rollout_limit must be positive when set")


@dataclass(frozen=True, slots=True)
class MCTSResult:
    """Search output for one root state."""

    action: int
    policy: np.ndarray
    visits: np.ndarray
    root: MCTSNode


def run_mcts(
    state: GameState,
    config: MCTSConfig | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> MCTSResult:
    """Run pure MCTS from state and return the most visited action."""
    config = MCTSConfig() if config is None else config
    rng = np.random.default_rng() if rng is None else rng
    if state.terminal:
        raise ValueError("cannot run MCTS from a terminal state")

    root = MCTSNode(
        player_to_move=state.player_to_move,
        untried_actions=_shuffled_actions(state.legal_actions(), rng),
    )

    for _ in range(config.simulations):
        simulation_state = state.copy(include_history=True)
        node = root
        path = [node]

        while not simulation_state.terminal and node.is_fully_expanded():
            node = node.best_child(config.exploration, rng)
            if node.action is None:
                raise RuntimeError("non-root MCTS child is missing an action")
            simulation_state.make_move(node.action)
            path.append(node)

        if not simulation_state.terminal and node.untried_actions:
            action = node.untried_actions.pop()
            simulation_state.make_move(action)
            child = MCTSNode(
                player_to_move=simulation_state.player_to_move,
                action=action,
                parent=node,
                untried_actions=_shuffled_actions(
                    simulation_state.legal_actions(),
                    rng,
                ),
            )
            node.children[action] = child
            node = child
            path.append(node)

        terminal_state = _rollout(simulation_state, config.rollout_limit, rng)
        _backup(path, terminal_state)

    visits = np.zeros(state.config.num_actions, dtype=np.float64)
    values = np.full(state.config.num_actions, -np.inf, dtype=np.float64)
    for action, child in root.children.items():
        visits[action] = child.visit_count
        values[action] = -child.mean_value

    legal_mask = state.legal_mask()
    policy = normalize_legal_policy(visits, legal_mask)
    action = _select_root_action(visits, values, legal_mask, rng)
    return MCTSResult(action=action, policy=policy, visits=visits, root=root)


def _rollout(
    state: GameState,
    rollout_limit: int | None,
    rng: np.random.Generator,
) -> GameState:
    steps = 0
    while not state.terminal and (rollout_limit is None or steps < rollout_limit):
        legal_actions = state.legal_actions()
        action = int(rng.choice(legal_actions))
        state.make_move(action)
        steps += 1
    return state


def _backup(path: list[MCTSNode], terminal_state: GameState) -> None:
    for node in path:
        node.visit_count += 1
        node.value_sum += _value_for_player(terminal_state, node.player_to_move)


def _value_for_player(state: GameState, player: int) -> float:
    if not state.terminal:
        return 0.0
    return float(state.outcome_for_player(player))


def _shuffled_actions(
    actions: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    shuffled = [int(action) for action in actions]
    rng.shuffle(shuffled)
    return shuffled


def _select_root_action(
    visits: np.ndarray,
    values: np.ndarray,
    legal_mask: np.ndarray,
    rng: np.random.Generator,
) -> int:
    legal_actions = np.flatnonzero(legal_mask)
    legal_visits = visits[legal_actions]
    max_visits = legal_visits.max()
    candidates = legal_actions[legal_visits == max_visits]
    if candidates.size == 1:
        return int(candidates[0])

    candidate_values = values[candidates]
    max_value = candidate_values.max()
    candidates = candidates[candidate_values == max_value]
    return int(rng.choice(candidates))

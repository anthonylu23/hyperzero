from dataclasses import dataclass

import numpy as np
import pytest
import torch

from hyperzero.agents import AlphaZeroAgent
from hyperzero.game import GameConfig, GameState, TerminalStateError
from hyperzero.models import NeuralEvaluator, PolicyValueMLP
from hyperzero.search import (
    PolicyValueEvaluation,
    PUCTConfig,
    PUCTNode,
    PUCTSearchSession,
    run_puct,
)


@dataclass(slots=True)
class FixedEvaluator:
    logits: np.ndarray
    value: float = 0.0

    def evaluate(self, state: GameState) -> PolicyValueEvaluation:
        return PolicyValueEvaluation(self.logits, self.value)

    def evaluate_many(self, states: list[GameState]) -> list[PolicyValueEvaluation]:
        return [self.evaluate(state) for state in states]


def test_puct_node_scores_child_value_from_parent_perspective() -> None:
    rng = np.random.default_rng(0)
    parent = PUCTNode(player_to_move=1, visit_count=2)
    bad_for_parent = PUCTNode(
        player_to_move=-1,
        action=0,
        parent=parent,
        visit_count=1,
        value_sum=1.0,
    )
    good_for_parent = PUCTNode(
        player_to_move=-1,
        action=1,
        parent=parent,
        visit_count=1,
        value_sum=-1.0,
    )
    parent.children = {0: bad_for_parent, 1: good_for_parent}

    assert parent.best_child(c_puct=0.0, rng=rng).action == 1


def test_run_puct_returns_legal_normalized_policy() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)
    state.make_move(0)
    state.make_move(0)
    evaluator = FixedEvaluator(np.array([1.0, 2.0, 20.0]))

    result = run_puct(
        state,
        evaluator,
        PUCTConfig(simulations=12, c_puct=1.5),
        rng=np.random.default_rng(0),
    )

    assert state.legal_mask()[result.action]
    np.testing.assert_allclose(result.policy.sum(), 1.0)
    assert np.all(result.policy[~state.legal_mask()] == 0.0)


def test_puct_search_session_runs_stepwise() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)
    evaluator = FixedEvaluator(np.zeros(config.num_actions))
    session = PUCTSearchSession(
        state,
        PUCTConfig(simulations=6, c_puct=1.5),
        np.random.default_rng(0),
    )

    leaf = session.select_leaf()
    session.complete_leaf(leaf, evaluator.evaluate(leaf.state))
    while not session.complete:
        leaf = session.select_leaf()
        evaluation = (
            evaluator.evaluate(leaf.state) if leaf.requires_evaluation else None
        )
        session.complete_leaf(leaf, evaluation)
    result = session.result()

    assert state.legal_mask()[result.action]
    assert result.root.visit_count == 6
    assert int(result.visits.sum()) == 6
    np.testing.assert_allclose(result.policy.sum(), 1.0)


def test_run_puct_can_prefer_immediate_win_over_uniform_evaluator() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)
    for action in [0, 1, 0, 1]:
        state.make_move(action)
    evaluator = FixedEvaluator(np.zeros(config.num_actions))

    result = run_puct(
        state,
        evaluator,
        PUCTConfig(simulations=20, c_puct=1.5),
        rng=np.random.default_rng(0),
    )

    assert result.action == 0


def test_run_puct_root_guard_blocks_immediate_loss() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)
    for action in [1, 0, 2, 0]:
        state.make_move(action)
    evaluator = FixedEvaluator(np.array([-10.0, 10.0, 9.0]))

    result = run_puct(
        state,
        evaluator,
        PUCTConfig(simulations=2, c_puct=1.5),
        rng=np.random.default_rng(0),
    )

    assert result.action == 0
    np.testing.assert_allclose(result.policy, np.array([1.0, 0.0, 0.0]))


def test_alphazero_agent_records_last_result() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)
    evaluator = FixedEvaluator(np.zeros(config.num_actions))
    agent = AlphaZeroAgent(evaluator, simulations=8, seed=0)

    action = agent.select_action(state)

    assert state.legal_mask()[action]
    assert agent.last_result is not None
    np.testing.assert_allclose(agent.last_result.policy.sum(), 1.0)


def test_alphazero_agent_runs_with_neural_evaluator() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)
    torch.manual_seed(0)
    model = PolicyValueMLP.from_config(config, hidden_size=16, residual_blocks=1)
    evaluator = NeuralEvaluator(model)
    agent = AlphaZeroAgent(evaluator, simulations=8, seed=0)

    action = agent.select_action(state)

    assert state.legal_mask()[action]
    assert agent.last_result is not None
    np.testing.assert_allclose(agent.last_result.policy.sum(), 1.0)
    assert np.all(agent.last_result.policy[~state.legal_mask()] == 0.0)
    assert agent.last_result.root.visit_count == 8
    assert int(agent.last_result.visits.sum()) == 8


def test_alphazero_agent_rejects_terminal_state() -> None:
    config = GameConfig(shape=(1, 1), connect_k=1, gravity_axis=0)
    state = GameState.new(config)
    state.make_move(0)
    evaluator = FixedEvaluator(np.zeros(config.num_actions))

    with pytest.raises(TerminalStateError):
        AlphaZeroAgent(evaluator).select_action(state)

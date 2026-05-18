import numpy as np
import pytest

from hyperzero.agents import HeuristicAgent, MCTSAgent, RandomAgent, TacticalAgent
from hyperzero.game import GameConfig, GameState, TerminalStateError


def test_random_agent_selects_only_legal_actions() -> None:
    config = GameConfig(shape=(2, 3), connect_k=2, gravity_axis=0)
    state = GameState.new(config)
    state.make_move(0)
    state.make_move(0)

    agent = RandomAgent(seed=0)
    actions = {agent.select_action(state) for _ in range(20)}

    assert actions <= {1, 2}


def test_random_agent_rejects_terminal_state() -> None:
    config = GameConfig(shape=(1, 1), connect_k=1, gravity_axis=0)
    state = GameState.new(config)
    state.make_move(0)

    with pytest.raises(TerminalStateError):
        RandomAgent(seed=0).select_action(state)


def test_tactical_agent_takes_immediate_win() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)
    for action in [0, 1, 0, 1]:
        state.make_move(action)

    assert TacticalAgent(seed=0).select_action(state) == 0


def test_tactical_agent_blocks_immediate_loss() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)
    for action in [1, 0, 2, 0]:
        state.make_move(action)

    assert TacticalAgent(seed=0).select_action(state) == 0


def test_heuristic_agent_takes_immediate_win() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)
    for action in [0, 1, 0, 1]:
        state.make_move(action)

    assert HeuristicAgent(seed=0).select_action(state) == 0


def test_heuristic_agent_prefers_blocking_major_threat() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)
    for action in [1, 0, 2, 0]:
        state.make_move(action)

    assert HeuristicAgent(seed=0).select_action(state) == 0


def test_mcts_agent_returns_legal_action_and_policy() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)
    agent = MCTSAgent(simulations=12, seed=0)

    action = agent.select_action(state)

    assert state.legal_mask()[action]
    assert agent.last_result is not None
    np.testing.assert_allclose(agent.last_result.policy.sum(), 1.0)
    assert np.all(agent.last_result.policy[~state.legal_mask()] == 0.0)

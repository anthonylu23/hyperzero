import json
from dataclasses import dataclass

import numpy as np

from hyperzero.game import GameConfig, GameReplay
from hyperzero.search import PolicyValueEvaluation, PUCTConfig
from hyperzero.training import (
    ReplayBuffer,
    TrainingConfig,
    generate_game,
    generate_games_batched,
    load_training_checkpoint,
    train_v1,
)


@dataclass(slots=True)
class UniformEvaluator:
    num_actions: int

    def evaluate(self, state) -> PolicyValueEvaluation:
        return PolicyValueEvaluation(np.zeros(self.num_actions), 0.0)

    def evaluate_many(self, states) -> list[PolicyValueEvaluation]:
        return [self.evaluate(state) for state in states]


def test_generate_game_returns_finalized_self_play_examples() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    game = generate_game(
        config,
        UniformEvaluator(config.num_actions),
        search_config=PUCTConfig(simulations=2),
        rng=np.random.default_rng(0),
    )
    replay = GameReplay.from_actions(config, game.actions)

    assert game.terminal
    assert game.winner == replay.winner
    assert len(game.examples) == len(game.actions)
    for example in game.examples:
        assert example.board.shape == (config.num_cells,)
        assert example.policy.shape == (config.num_actions,)
        np.testing.assert_allclose(example.policy.sum(), 1.0)
        assert np.all(example.policy[~example.legal_mask] == 0.0)
        assert example.value in (-1.0, 0.0, 1.0)


def test_generate_games_batched_returns_valid_games() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    games = generate_games_batched(
        config,
        UniformEvaluator(config.num_actions),
        games=3,
        search_config=PUCTConfig(simulations=2),
        rng=np.random.default_rng(0),
        max_active_games=2,
    )

    assert len(games) == 3
    for game in games:
        assert game.terminal
        assert len(game.examples) == len(game.actions)
        for example in game.examples:
            np.testing.assert_allclose(example.policy.sum(), 1.0)
            assert np.all(example.policy[~example.legal_mask] == 0.0)


def test_replay_buffer_evicts_old_examples_and_samples() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    game = generate_game(
        config,
        UniformEvaluator(config.num_actions),
        search_config=PUCTConfig(simulations=1),
        rng=np.random.default_rng(1),
    )
    buffer = ReplayBuffer(capacity=3, seed=0)

    buffer.add_many(game.examples)
    sample = buffer.sample(10)

    assert len(buffer) == 3
    assert len(sample) == 3


def test_train_v1_runs_one_iteration_and_writes_checkpoint(tmp_path) -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)

    result = train_v1(
        TrainingConfig(
            game_config=config,
            iterations=1,
            self_play_games_per_iteration=1,
            puct_simulations=1,
            training_steps_per_iteration=1,
            batch_size=4,
            hidden_size=8,
            residual_blocks=0,
            seed=0,
            checkpoint_dir=tmp_path,
            eval_games_per_iteration=1,
            eval_opponents=("random",),
            eval_simulations=1,
        )
    )

    assert len(result.metrics) == 1
    metric = result.metrics[0]
    assert metric.self_play_games == 1
    assert metric.self_play_examples > 0
    assert metric.replay_size == metric.self_play_examples
    assert metric.total_loss > 0.0
    assert "random" in metric.evaluations
    assert metric.checkpoint_path is not None
    assert tmp_path.joinpath("iteration_0001.pt").exists()
    assert tmp_path.joinpath("metrics.jsonl").exists()
    logged_metric = json.loads(tmp_path.joinpath("metrics.jsonl").read_text())
    assert logged_metric["policy_loss"] == metric.policy_loss
    assert logged_metric["policy_loss_min"] <= logged_metric["policy_loss"]
    assert logged_metric["policy_loss_max"] >= logged_metric["policy_loss"]
    assert logged_metric["value_loss"] == metric.value_loss
    assert logged_metric["value_loss_min"] <= logged_metric["value_loss"]
    assert logged_metric["value_loss_max"] >= logged_metric["value_loss"]
    assert logged_metric["total_loss"] == metric.total_loss
    assert logged_metric["total_loss_min"] <= logged_metric["total_loss"]
    assert logged_metric["total_loss_max"] >= logged_metric["total_loss"]
    assert logged_metric["training_steps"] == 1
    assert logged_metric["batch_size"] == 4
    assert logged_metric["puct_simulations"] == 1
    assert logged_metric["eval_games"] == 1
    assert logged_metric["eval_opponents"] == ["random"]
    assert logged_metric["eval_simulations"] == 1
    assert "agent_a_win_rate" in logged_metric["evaluations"]["random"]

    checkpoint = load_training_checkpoint(metric.checkpoint_path)
    assert checkpoint.iteration == 1
    assert checkpoint.game_config.shape == config.shape


def test_train_v1_runs_batched_self_play(tmp_path) -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)

    result = train_v1(
        TrainingConfig(
            game_config=config,
            iterations=1,
            self_play_games_per_iteration=2,
            puct_simulations=1,
            training_steps_per_iteration=1,
            batch_size=4,
            hidden_size=8,
            residual_blocks=0,
            seed=0,
            checkpoint_dir=tmp_path,
            batched_self_play=True,
            max_active_self_play_games=2,
        )
    )

    assert len(result.metrics) == 1
    assert result.metrics[0].batched_self_play
    assert result.metrics[0].max_active_self_play_games == 2
    assert result.metrics[0].self_play_games == 2
    assert result.metrics[0].self_play_examples > 0
    logged_metric = json.loads(tmp_path.joinpath("metrics.jsonl").read_text())
    assert logged_metric["evaluations"] == {}
    assert logged_metric["eval_games"] == 0
    assert logged_metric["eval_opponents"] == []
    assert logged_metric["batched_self_play"] is True
    assert logged_metric["max_active_self_play_games"] == 2

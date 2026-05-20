import json

import numpy as np
import torch

from hyperzero.agents import AlphaZeroAgent
from hyperzero.game import GameConfig, GameState
from hyperzero.models import (
    UniversalEvaluator,
    UniversalModelConfig,
    UniversalPolicyValueTransformer,
)
from hyperzero.training import (
    UniversalGameSpec,
    UniversalReplayBuffer,
    UniversalTrainingConfig,
    build_universal_checkpoint_agent,
    load_universal_training_checkpoint,
    train_universal,
)
from hyperzero.training.universal_replay import UniversalSelfPlayExample
from hyperzero.universal import (
    UniversalEncoderConfig,
    collate_positions,
    encode_state,
)


def test_universal_encoding_collates_mixed_ranks() -> None:
    configs = (
        GameConfig(shape=(4, 4), connect_k=3, gravity_axis=0),
        GameConfig(shape=(4, 4, 4), connect_k=4, gravity_axis=0),
        GameConfig(shape=(4, 4, 4, 4), connect_k=4, gravity_axis=0),
    )
    positions = [encode_state(GameState.new(config)) for config in configs]

    batch = collate_positions(positions)

    assert batch.cell_features.shape == (3, 256, UniversalEncoderConfig().feature_size)
    assert batch.action_features.shape == (3, 64, UniversalEncoderConfig().feature_size)
    assert batch.cell_mask.sum().item() == 16 + 64 + 256
    assert batch.action_mask.sum().item() == 4 + 16 + 64


def test_universal_model_and_evaluator_return_variant_action_shapes() -> None:
    configs = (
        GameConfig(shape=(4, 4), connect_k=3, gravity_axis=0),
        GameConfig(shape=(4, 4, 4), connect_k=4, gravity_axis=0),
        GameConfig(shape=(4, 4, 4, 4), connect_k=4, gravity_axis=0),
    )
    model_config = UniversalModelConfig(hidden_size=16, residual_blocks=1, heads=4)
    model = UniversalPolicyValueTransformer(model_config)
    evaluator = UniversalEvaluator(model, model_config.encoder)

    evaluations = evaluator.evaluate_many([GameState.new(config) for config in configs])

    assert [evaluation.policy_logits.shape for evaluation in evaluations] == [
        (4,),
        (16,),
        (64,),
    ]
    assert all(-1.0 <= evaluation.value <= 1.0 for evaluation in evaluations)


def test_universal_agent_can_select_legal_action_across_variants() -> None:
    model_config = UniversalModelConfig(hidden_size=16, residual_blocks=1, heads=4)
    model = UniversalPolicyValueTransformer(model_config)
    agent = AlphaZeroAgent(
        UniversalEvaluator(model, model_config.encoder),
        simulations=1,
        seed=0,
    )
    for config in (
        GameConfig(shape=(4, 4), connect_k=3, gravity_axis=0),
        GameConfig(shape=(4, 4, 4), connect_k=4, gravity_axis=0),
        GameConfig(shape=(4, 4, 4, 4), connect_k=4, gravity_axis=0),
    ):
        state = GameState.new(config)
        action = agent.select_action(state)

        assert state.legal_mask()[action]


def test_universal_replay_samples_balanced_configs() -> None:
    config = GameConfig(shape=(4, 4), connect_k=3, gravity_axis=0)
    buffer = UniversalReplayBuffer(capacity=10, seed=0)
    for index, config_id in enumerate(("a", "b", "a", "b")):
        buffer.add(
            UniversalSelfPlayExample(
                config_id=config_id,
                game_config=config,
                board=np.zeros(config.num_cells, dtype=np.float32),
                policy=np.full(config.num_actions, 1.0 / config.num_actions),
                value=0.0,
                legal_mask=np.ones(config.num_actions, dtype=bool),
                player_to_move=1,
                ply=index,
            )
        )

    sample = buffer.sample(4, balanced=True)

    assert {example.config_id for example in sample} == {"a", "b"}
    assert buffer.counts_by_config() == {"a": 2, "b": 2}


def test_train_universal_runs_and_writes_loadable_checkpoint(tmp_path) -> None:
    result = train_universal(
        UniversalTrainingConfig(
            game_specs=(
                UniversalGameSpec(
                    "2d",
                    GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0),
                    1,
                ),
                UniversalGameSpec(
                    "3d",
                    GameConfig(shape=(3, 3, 3), connect_k=3, gravity_axis=0),
                    1,
                ),
            ),
            iterations=1,
            puct_simulations=1,
            training_steps_per_iteration=1,
            batch_size=4,
            replay_capacity=64,
            hidden_size=16,
            residual_blocks=1,
            heads=4,
            seed=0,
            checkpoint_dir=tmp_path,
            eval_games_per_variant=1,
            eval_opponents=("random",),
            eval_simulations=1,
        )
    )

    assert len(result.metrics) == 1
    metric = result.metrics[0]
    assert metric.self_play_games_by_config == {"2d": 1, "3d": 1}
    assert metric.self_play_examples_by_config["2d"] > 0
    assert metric.self_play_examples_by_config["3d"] > 0
    assert metric.replay_size == metric.self_play_examples
    assert metric.total_loss > 0.0
    assert "2d" in metric.evaluations
    assert "random" in metric.evaluations["2d"]
    assert metric.checkpoint_path is not None
    assert tmp_path.joinpath("iteration_0001.pt").exists()
    assert tmp_path.joinpath("best_by_eval_score.pt").exists()

    logged_metric = json.loads(tmp_path.joinpath("metrics.jsonl").read_text())
    assert logged_metric["self_play_games_by_config"] == {"2d": 1, "3d": 1}
    assert logged_metric["eval_score"] is not None

    checkpoint = load_universal_training_checkpoint(metric.checkpoint_path)
    assert checkpoint.iteration == 1
    assert len(checkpoint.game_specs) == 2
    agent, loaded = build_universal_checkpoint_agent(
        metric.checkpoint_path,
        simulations=1,
    )
    state = GameState.new(loaded.game_specs[0].game_config)
    assert state.legal_mask()[agent.select_action(state)]


def test_train_universal_resumes_from_checkpoint(tmp_path) -> None:
    spec = UniversalGameSpec(
        "2d",
        GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0),
        1,
    )
    first = train_universal(
        UniversalTrainingConfig(
            game_specs=(spec,),
            iterations=1,
            puct_simulations=1,
            training_steps_per_iteration=1,
            batch_size=4,
            replay_capacity=64,
            hidden_size=16,
            residual_blocks=1,
            seed=0,
            checkpoint_dir=tmp_path,
            checkpoint_keep_last=2,
        )
    )
    checkpoint_path = first.metrics[0].checkpoint_path
    assert checkpoint_path is not None

    resumed = train_universal(
        UniversalTrainingConfig(
            game_specs=(spec,),
            iterations=2,
            puct_simulations=1,
            training_steps_per_iteration=1,
            batch_size=4,
            replay_capacity=64,
            hidden_size=16,
            residual_blocks=1,
            seed=0,
            checkpoint_dir=tmp_path,
            checkpoint_keep_last=2,
            resume_from_checkpoint=checkpoint_path,
        )
    )

    assert len(resumed.metrics) == 1
    assert resumed.metrics[0].iteration == 2
    assert resumed.metrics[0].replay_size > first.metrics[0].replay_size
    assert tmp_path.joinpath("iteration_0002.pt").exists()
    raw = torch.load(
        tmp_path / "iteration_0002.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert raw["checkpoint_version"] == 2
    assert "replay_buffer" in raw
    assert len(raw["metrics"]) == 2


def test_train_universal_resume_allows_curriculum_count_change(tmp_path) -> None:
    first_spec = UniversalGameSpec(
        "2d",
        GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0),
        1,
    )
    first = train_universal(
        UniversalTrainingConfig(
            game_specs=(first_spec,),
            iterations=1,
            puct_simulations=1,
            training_steps_per_iteration=1,
            batch_size=4,
            replay_capacity=64,
            hidden_size=16,
            residual_blocks=1,
            seed=0,
            checkpoint_dir=tmp_path,
            checkpoint_keep_last=2,
        )
    )
    checkpoint_path = first.metrics[0].checkpoint_path
    assert checkpoint_path is not None

    resumed = train_universal(
        UniversalTrainingConfig(
            game_specs=(
                UniversalGameSpec(
                    "2d",
                    GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0),
                    2,
                ),
            ),
            iterations=2,
            puct_simulations=1,
            training_steps_per_iteration=1,
            batch_size=4,
            replay_capacity=64,
            hidden_size=16,
            residual_blocks=1,
            seed=0,
            checkpoint_dir=tmp_path,
            checkpoint_keep_last=2,
            resume_from_checkpoint=checkpoint_path,
        )
    )

    assert resumed.metrics[0].self_play_games_by_config == {"2d": 2}


def test_train_universal_eval_floors_gate_best_checkpoint(tmp_path) -> None:
    result = train_universal(
        UniversalTrainingConfig(
            game_specs=(
                UniversalGameSpec(
                    "2d",
                    GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0),
                    1,
                ),
            ),
            iterations=1,
            puct_simulations=1,
            training_steps_per_iteration=1,
            batch_size=4,
            replay_capacity=64,
            hidden_size=16,
            residual_blocks=1,
            seed=0,
            checkpoint_dir=tmp_path,
            eval_games_per_variant=1,
            eval_opponents=("random",),
            eval_simulations=1,
            eval_score_floors={"default": {"mcts": 1.0}},
        )
    )

    metric = result.metrics[0]
    assert metric.eval_score is not None
    assert metric.eval_floor_passed is False
    assert not metric.is_best_checkpoint
    assert metric.eval_floor_failures == ("2d:mcts=missing<1.000",)
    assert not tmp_path.joinpath("best_by_eval_score.pt").exists()

import json

import numpy as np
import pytest
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
from hyperzero.training.train_universal import _sample_training_batch
from hyperzero.training.universal_replay import UniversalSelfPlayExample
from hyperzero.universal import (
    UniversalEncoderConfig,
    collate_positions,
    encode_position,
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


def test_universal_line_features_expose_landing_threats() -> None:
    config = GameConfig(shape=(4, 4), connect_k=3, gravity_axis=0)
    board = np.zeros(config.num_cells, dtype=np.float32)
    board[config.flat_index((0, 0))] = 1.0
    board[config.flat_index((1, 0))] = 1.0
    board[config.flat_index((0, 1))] = -1.0
    board[config.flat_index((1, 1))] = -1.0
    encoder = UniversalEncoderConfig(line_features=True)

    position = encode_position(
        config,
        board=board,
        legal_mask=np.ones(config.num_actions, dtype=bool),
        ply=4,
        encoder_config=encoder,
    )

    line_offset = UniversalEncoderConfig().feature_size
    assert encoder.feature_size == line_offset + 4
    assert position.action_features[0, line_offset + 2] == pytest.approx(1.0)
    assert position.action_features[1, line_offset + 3] == pytest.approx(2 / 3)
    assert position.cell_features[config.flat_index((0, 0)), line_offset] > 0.0
    assert position.cell_features[config.flat_index((0, 1)), line_offset + 1] > 0.0


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


def test_universal_model_with_line_features_and_rank_adapters() -> None:
    configs = (
        GameConfig(shape=(4, 4), connect_k=3, gravity_axis=0),
        GameConfig(shape=(4, 4, 4), connect_k=4, gravity_axis=0),
    )
    model_config = UniversalModelConfig(
        encoder=UniversalEncoderConfig(line_features=True),
        hidden_size=16,
        residual_blocks=1,
        heads=4,
        input_layer_norm=True,
        rank_adapters=True,
        rank_head_adapters=True,
        adapter_size=8,
    )
    model = UniversalPolicyValueTransformer(model_config)
    evaluator = UniversalEvaluator(model, model_config.encoder)

    evaluations = evaluator.evaluate_many([GameState.new(config) for config in configs])

    assert [evaluation.policy_logits.shape for evaluation in evaluations] == [
        (4,),
        (16,),
    ]
    assert all(-1.0 <= evaluation.value <= 1.0 for evaluation in evaluations)
    assert UniversalModelConfig.from_dict(model_config.to_dict()) == model_config


def test_universal_model_rank_head_adapters_start_as_zero_delta() -> None:
    model_config = UniversalModelConfig(
        hidden_size=16,
        residual_blocks=1,
        heads=4,
        rank_head_adapters=True,
        adapter_size=8,
    )
    base_model = UniversalPolicyValueTransformer(
        UniversalModelConfig(hidden_size=16, residual_blocks=1, heads=4)
    )
    adapted_model = UniversalPolicyValueTransformer(model_config)
    missing = adapted_model.load_state_dict(base_model.state_dict(), strict=False)
    assert all(
        key.startswith(("policy_rank_head_adapters.", "value_rank_head_adapters."))
        for key in missing.missing_keys
    )
    assert missing.unexpected_keys == []

    positions = [
        encode_state(
            GameState.new(GameConfig(shape=(4, 4), connect_k=3, gravity_axis=0))
        ),
        encode_state(
            GameState.new(GameConfig(shape=(4, 4, 4), connect_k=4, gravity_axis=0))
        ),
    ]
    batch = collate_positions(positions)

    base_policy, base_value = base_model(batch)
    adapted_policy, adapted_value = adapted_model(batch)

    assert torch.allclose(adapted_policy, base_policy)
    assert torch.allclose(adapted_value, base_value)


def test_universal_model_line_policy_residual_starts_as_zero_delta() -> None:
    encoder = UniversalEncoderConfig(line_features=True)
    base_config = UniversalModelConfig(
        encoder=encoder,
        hidden_size=16,
        residual_blocks=1,
        heads=4,
    )
    residual_config = UniversalModelConfig(
        encoder=encoder,
        hidden_size=16,
        residual_blocks=1,
        heads=4,
        line_policy_residual=True,
        adapter_size=8,
    )
    base_model = UniversalPolicyValueTransformer(base_config)
    residual_model = UniversalPolicyValueTransformer(residual_config)
    missing = residual_model.load_state_dict(base_model.state_dict(), strict=False)
    assert all(key.startswith("line_policy_residual.") for key in missing.missing_keys)
    assert missing.unexpected_keys == []

    positions = [
        encode_position(
            GameConfig(shape=(4, 4), connect_k=3, gravity_axis=0),
            board=np.zeros(16, dtype=np.float32),
            legal_mask=np.ones(4, dtype=bool),
            ply=0,
            encoder_config=encoder,
        )
    ]
    batch = collate_positions(positions)

    base_policy, base_value = base_model(batch)
    residual_policy, residual_value = residual_model(batch)

    assert torch.allclose(residual_policy, base_policy)
    assert torch.allclose(residual_value, base_value)


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


def test_teacher_replay_sampling_mixes_teacher_fraction() -> None:
    config = GameConfig(shape=(4, 4), connect_k=3, gravity_axis=0)
    main = UniversalReplayBuffer(capacity=8, seed=0)
    teacher = UniversalReplayBuffer(capacity=8, seed=1)
    for index in range(4):
        main.add(
            UniversalSelfPlayExample(
                config_id="main",
                game_config=config,
                board=np.zeros(config.num_cells, dtype=np.float32),
                policy=np.full(config.num_actions, 1.0 / config.num_actions),
                value=0.0,
                legal_mask=np.ones(config.num_actions, dtype=bool),
                player_to_move=1,
                ply=index,
            )
        )
        teacher.add(
            UniversalSelfPlayExample(
                config_id="teacher",
                game_config=config,
                board=np.zeros(config.num_cells, dtype=np.float32),
                policy=np.full(config.num_actions, 1.0 / config.num_actions),
                value=0.0,
                legal_mask=np.ones(config.num_actions, dtype=bool),
                player_to_move=1,
                ply=index,
            )
        )

    batch = _sample_training_batch(
        main,
        teacher,
        batch_size=4,
        teacher_batch_fraction=0.5,
        balanced=True,
    )

    assert [example.config_id for example in batch].count("teacher") == 2
    assert [example.config_id for example in batch].count("main") == 2


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


def test_train_universal_can_reset_optimizer_on_resume(tmp_path) -> None:
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
            learning_rate=1e-3,
            hidden_size=16,
            residual_blocks=1,
            seed=0,
            checkpoint_dir=tmp_path,
            checkpoint_keep_last=2,
        )
    )
    checkpoint_path = first.metrics[0].checkpoint_path
    assert checkpoint_path is not None

    train_universal(
        UniversalTrainingConfig(
            game_specs=(spec,),
            iterations=2,
            puct_simulations=1,
            training_steps_per_iteration=1,
            batch_size=4,
            replay_capacity=64,
            learning_rate=1e-4,
            hidden_size=16,
            residual_blocks=1,
            seed=0,
            checkpoint_dir=tmp_path,
            checkpoint_keep_last=2,
            resume_from_checkpoint=checkpoint_path,
            reset_optimizer_on_resume=True,
        )
    )

    raw = torch.load(
        tmp_path / "iteration_0002.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert raw["optimizer_state_dict"]["param_groups"][0]["lr"] == pytest.approx(1e-4)
    assert raw["training_config"]["reset_optimizer_on_resume"] is True


def test_train_universal_can_reset_replay_and_rng_on_resume(tmp_path) -> None:
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
            seed=123,
            checkpoint_dir=tmp_path,
            checkpoint_keep_last=2,
            resume_from_checkpoint=checkpoint_path,
            reset_replay_on_resume=True,
            reset_rng_on_resume=True,
        )
    )

    metric = resumed.metrics[0]
    assert metric.iteration == 2
    assert metric.replay_size == metric.self_play_examples
    assert metric.replay_size <= first.metrics[0].replay_size

    raw = torch.load(
        tmp_path / "iteration_0002.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert len(raw["replay_buffer"]["examples"]) == metric.self_play_examples
    assert raw["training_config"]["reset_replay_on_resume"] is True
    assert raw["training_config"]["reset_rng_on_resume"] is True


def test_train_universal_can_add_rank_head_adapters_on_resume(tmp_path) -> None:
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
            rank_head_adapters=True,
            adapter_size=8,
            seed=0,
            checkpoint_dir=tmp_path,
            checkpoint_keep_last=2,
            resume_from_checkpoint=checkpoint_path,
            reset_optimizer_on_resume=True,
        )
    )

    assert resumed.metrics[0].iteration == 2
    raw = torch.load(
        tmp_path / "iteration_0002.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert raw["universal_model_config"]["rank_head_adapters"] is True
    assert raw["training_config"]["rank_head_adapters"] is True
    assert any(
        key.startswith("policy_rank_head_adapters.") for key in raw["model_state_dict"]
    )


def test_train_universal_can_add_line_policy_residual_on_resume(tmp_path) -> None:
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
            line_features=True,
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
            line_features=True,
            line_policy_residual=True,
            adapter_size=8,
            seed=0,
            checkpoint_dir=tmp_path,
            checkpoint_keep_last=2,
            resume_from_checkpoint=checkpoint_path,
            reset_optimizer_on_resume=True,
        )
    )

    assert resumed.metrics[0].iteration == 2
    raw = torch.load(
        tmp_path / "iteration_0002.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert raw["universal_model_config"]["line_policy_residual"] is True
    assert raw["training_config"]["line_policy_residual"] is True
    assert any(
        key.startswith("line_policy_residual.") for key in raw["model_state_dict"]
    )


def test_train_universal_uses_teacher_replay_path(tmp_path) -> None:
    game_config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    teacher_example = UniversalSelfPlayExample(
        config_id="2d",
        game_config=game_config,
        board=np.zeros(game_config.num_cells, dtype=np.float32),
        policy=np.full(game_config.num_actions, 1.0 / game_config.num_actions),
        value=0.0,
        legal_mask=np.ones(game_config.num_actions, dtype=bool),
        player_to_move=1,
        ply=0,
    )
    teacher_path = tmp_path / "teacher.pt"
    torch.save(
        {
            "format": "universal_teacher_replay_v1",
            "examples": [teacher_example],
        },
        teacher_path,
    )

    result = train_universal(
        UniversalTrainingConfig(
            game_specs=(UniversalGameSpec("2d", game_config, 1),),
            iterations=1,
            puct_simulations=1,
            training_steps_per_iteration=1,
            batch_size=4,
            replay_capacity=64,
            hidden_size=16,
            residual_blocks=1,
            seed=0,
            checkpoint_dir=tmp_path,
            teacher_replay_path=teacher_path,
            teacher_batch_fraction=0.5,
        )
    )

    metric = result.metrics[0]
    assert metric.teacher_replay_size == 1
    assert metric.teacher_batch_fraction == pytest.approx(0.5)
    raw = torch.load(
        tmp_path / "iteration_0001.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert raw["training_config"]["teacher_replay_path"] == str(teacher_path)
    assert raw["training_config"]["teacher_batch_fraction"] == pytest.approx(0.5)


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
    assert tmp_path.joinpath("best_current_run_raw.pt").exists()
    assert not tmp_path.joinpath("best_current_run_floor_passing.pt").exists()


def test_train_universal_parallel_workers_run(tmp_path) -> None:
    result = train_universal(
        UniversalTrainingConfig(
            game_specs=(
                UniversalGameSpec(
                    "2d_a",
                    GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0),
                    1,
                ),
                UniversalGameSpec(
                    "2d_b",
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
            eval_workers=2,
            self_play_workers=2,
            batched_self_play=True,
            max_active_self_play_games=1,
        )
    )

    metric = result.metrics[0]
    assert metric.self_play_workers == 2
    assert metric.eval_workers == 2
    assert metric.self_play_games_by_config == {"2d_a": 1, "2d_b": 1}
    assert set(metric.evaluations) == {"2d_a", "2d_b"}
    assert all("random" in stats for stats in metric.evaluations.values())
    assert metric.self_play_inference_batches > 0
    assert metric.eval_inference_batches > 0


def test_train_universal_central_batched_self_play_runs(tmp_path) -> None:
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
            seed=0,
            checkpoint_dir=tmp_path,
            batched_self_play=True,
            max_active_self_play_games=2,
            central_batched_self_play=True,
        )
    )

    metric = result.metrics[0]
    assert metric.central_batched_self_play
    assert metric.self_play_games_by_config == {"2d": 1, "3d": 1}
    assert metric.self_play_examples_by_config["2d"] > 0
    assert metric.self_play_examples_by_config["3d"] > 0
    assert metric.self_play_inference_batches > 0
    assert metric.self_play_inference_states >= metric.self_play_inference_batches


def test_train_universal_queued_worker_self_play_runs(tmp_path) -> None:
    result = train_universal(
        UniversalTrainingConfig(
            game_specs=(
                UniversalGameSpec(
                    "2d_a",
                    GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0),
                    1,
                ),
                UniversalGameSpec(
                    "2d_b",
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
            batched_self_play=True,
            max_active_self_play_games=1,
            self_play_workers=2,
            central_batched_self_play=True,
        )
    )

    metric = result.metrics[0]
    assert metric.self_play_workers == 2
    assert metric.central_batched_self_play
    assert metric.self_play_games_by_config == {"2d_a": 1, "2d_b": 1}
    assert metric.self_play_inference_batches > 0
    assert metric.self_play_inference_states > 0

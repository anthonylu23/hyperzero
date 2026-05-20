import numpy as np
import torch

from hyperzero.game import GameConfig, GameState
from hyperzero.models import NeuralEvaluator, PolicyValueMLP, build_policy_value_model
from hyperzero.models.line_mlp import PolicyValueLineMLP


def test_policy_value_mlp_matches_configured_shapes() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    model = PolicyValueMLP.from_config(config, hidden_size=16, residual_blocks=1)
    board = torch.zeros((2, config.num_cells))

    policy_logits, value = model(board)

    assert policy_logits.shape == (2, config.num_actions)
    assert value.shape == (2,)
    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)


def test_policy_value_mlp_accepts_single_board() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    model = PolicyValueMLP.from_config(config, hidden_size=16, residual_blocks=0)
    board = torch.zeros(config.num_cells)

    policy_logits, value = model(board)

    assert policy_logits.shape == (config.num_actions,)
    assert value.shape == ()


def test_policy_value_model_variants_match_configured_shapes() -> None:
    configs_by_model = {
        "mlp": GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0),
        "line_mlp": GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0),
        "cnn": GameConfig(shape=(4, 4, 4), connect_k=4, gravity_axis=0),
        "resnet": GameConfig(shape=(4, 4, 4), connect_k=4, gravity_axis=0),
        "line_resnet": GameConfig(shape=(4, 4, 4), connect_k=4, gravity_axis=0),
        "transformer": GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0),
    }
    for model_type, config in configs_by_model.items():
        model = build_policy_value_model(
            config,
            model_type=model_type,
            hidden_size=16,
            residual_blocks=1,
        )
        board = torch.zeros((2, config.num_cells))

        policy_logits, value = model(board)

        assert policy_logits.shape == (2, config.num_actions)
        assert value.shape == (2,)


def test_line_mlp_empty_board_line_features_are_zero() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    model = PolicyValueLineMLP.from_config(config, hidden_size=16, residual_blocks=0)
    board = torch.zeros((1, config.num_cells))

    features = model._features(board)

    assert torch.all(features == 0.0)


def test_neural_evaluator_returns_single_state_numpy_outputs() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)
    model = PolicyValueMLP.from_config(config, hidden_size=16, residual_blocks=0)
    evaluator = NeuralEvaluator(model)

    evaluation = evaluator.evaluate(state)

    assert evaluation.policy_logits.shape == (config.num_actions,)
    assert evaluation.policy_logits.dtype == np.float64
    assert -1.0 <= evaluation.value <= 1.0
    assert evaluator.inference_batches == 1
    assert evaluator.inference_states == 1
    assert evaluator.inference_time_seconds >= 0.0


def test_neural_evaluator_batched_outputs_match_single_outputs() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    first = GameState.new(config)
    second = GameState.new(config)
    second.make_move(0)
    model = PolicyValueMLP.from_config(config, hidden_size=16, residual_blocks=0)
    evaluator = NeuralEvaluator(model)

    single = [evaluator.evaluate(first), evaluator.evaluate(second)]
    batched = evaluator.evaluate_many([first, second])

    for single_eval, batched_eval in zip(single, batched, strict=True):
        np.testing.assert_allclose(
            single_eval.policy_logits,
            batched_eval.policy_logits,
            atol=1e-7,
        )
        np.testing.assert_allclose(single_eval.value, batched_eval.value, atol=1e-7)
    assert evaluator.inference_batches == 3
    assert evaluator.inference_states == 4
    assert evaluator.inference_time_seconds >= 0.0

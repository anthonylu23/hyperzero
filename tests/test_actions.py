import numpy as np
import pytest

from hyperzero.game import GameConfig
from hyperzero.game.actions import (
    action_tensor_to_policy,
    logits_to_policy,
    mask_logits,
    normalize_legal_policy,
    policy_to_action_tensor,
    sample_action,
)


def test_mask_logits_and_softmax_policy_ignore_illegal_actions() -> None:
    logits = np.array([1.0, 2.0, 10.0])
    legal_mask = np.array([True, True, False])

    masked = mask_logits(logits, legal_mask)
    policy = logits_to_policy(logits, legal_mask)

    assert np.isneginf(masked[2])
    assert policy[2] == 0.0
    assert np.isclose(policy.sum(), 1.0)
    assert policy[1] > policy[0]


def test_normalize_legal_policy_falls_back_to_uniform_legal() -> None:
    policy = normalize_legal_policy(
        np.array([0.0, 0.0, 5.0]),
        np.array([True, True, False]),
    )

    np.testing.assert_allclose(policy, np.array([0.5, 0.5, 0.0]))


def test_logits_to_policy_falls_back_when_legal_logits_are_negative_infinity() -> None:
    policy = logits_to_policy(
        np.array([-np.inf, -np.inf, 10.0]),
        np.array([True, True, False]),
    )

    np.testing.assert_allclose(policy, np.array([0.5, 0.5, 0.0]))


def test_policy_action_tensor_round_trip() -> None:
    config = GameConfig(shape=(4, 2, 3), connect_k=4, gravity_axis=0)
    policy = np.arange(config.num_actions, dtype=np.float64)

    tensor = policy_to_action_tensor(policy, config)
    restored = action_tensor_to_policy(tensor, config)

    assert tensor.shape == (2, 3)
    np.testing.assert_array_equal(restored, policy)


def test_sample_action_respects_legal_mask() -> None:
    rng = np.random.default_rng(0)
    samples = {
        sample_action(
            np.array([0.0, 0.0, 1.0]),
            np.array([False, True, False]),
            rng=rng,
        )
        for _ in range(10)
    }

    assert samples == {1}


def test_policy_without_legal_mass_rejects_empty_mask() -> None:
    with pytest.raises(ValueError):
        normalize_legal_policy(np.array([1.0]), np.array([False]))

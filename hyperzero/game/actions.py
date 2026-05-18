"""Action and policy helpers for flat Connect-K action spaces."""

from __future__ import annotations

from operator import index

import numpy as np

from hyperzero.game.config import GameConfig


def coerce_policy(policy: np.ndarray, config: GameConfig) -> np.ndarray:
    """Return a flat float policy vector with one value per action."""
    policy = np.asarray(policy, dtype=np.float64)
    if policy.shape == config.action_shape:
        return policy.reshape(config.num_actions)
    if policy.shape == (config.num_actions,):
        return policy.copy()
    raise ValueError(
        f"policy shape must be {config.action_shape} or ({config.num_actions},)"
    )


def mask_logits(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    *,
    invalid_value: float = -np.inf,
) -> np.ndarray:
    """Return logits with illegal actions replaced by invalid_value."""
    logits = np.asarray(logits, dtype=np.float64)
    legal_mask = np.asarray(legal_mask, dtype=bool)
    if logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask must have the same shape")

    masked = logits.copy()
    masked[~legal_mask] = invalid_value
    return masked


def normalize_legal_policy(policy: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Zero illegal actions and normalize legal probability mass."""
    policy = np.asarray(policy, dtype=np.float64)
    legal_mask = np.asarray(legal_mask, dtype=bool)
    if policy.shape != legal_mask.shape:
        raise ValueError("policy and legal_mask must have the same shape")
    if not legal_mask.any():
        raise ValueError("cannot normalize policy with no legal actions")

    normalized = np.where(legal_mask, policy, 0.0)
    total = float(normalized.sum())
    if total <= 0.0 or not np.isfinite(total):
        normalized = legal_mask.astype(np.float64)
        total = float(normalized.sum())
    return normalized / total


def logits_to_policy(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Convert logits to a normalized policy over legal actions."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    logits = mask_logits(np.asarray(logits, dtype=np.float64), legal_mask)
    if not np.asarray(legal_mask, dtype=bool).any():
        raise ValueError("cannot build policy with no legal actions")

    scaled = logits / temperature
    legal_mask = np.asarray(legal_mask, dtype=bool)
    legal_values = scaled[legal_mask & np.isfinite(scaled)]
    if legal_values.size == 0:
        return normalize_legal_policy(np.zeros_like(scaled), legal_mask)
    shifted = scaled - np.max(legal_values)
    exp = np.exp(shifted)
    exp[~legal_mask] = 0.0
    return normalize_legal_policy(exp, legal_mask)


def policy_to_action_tensor(policy: np.ndarray, config: GameConfig) -> np.ndarray:
    """Return policy probabilities shaped like the non-gravity action grid."""
    return coerce_policy(policy, config).reshape(config.action_shape)


def action_tensor_to_policy(action_tensor: np.ndarray, config: GameConfig) -> np.ndarray:
    """Flatten an action-grid tensor into policy/action order."""
    action_tensor = np.asarray(action_tensor)
    if action_tensor.shape != config.action_shape:
        raise ValueError(f"action tensor shape must be {config.action_shape}")
    return action_tensor.reshape(config.num_actions).copy()


def sample_action(
    policy: np.ndarray,
    legal_mask: np.ndarray | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample one action id from a policy, optionally renormalizing over legal moves."""
    rng = np.random.default_rng() if rng is None else rng
    policy = np.asarray(policy, dtype=np.float64)
    if legal_mask is not None:
        policy = normalize_legal_policy(policy, np.asarray(legal_mask, dtype=bool))
    else:
        total = float(policy.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError("policy must contain positive finite probability mass")
        policy = policy / total
    return int(rng.choice(np.arange(policy.size), p=policy))


def validate_action(action: int, config: GameConfig) -> int:
    """Return action as a strict integer index, or raise ValueError."""
    action = index(action)
    config.validate_action_index(action)
    return action

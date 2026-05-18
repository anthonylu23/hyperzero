"""Core N-dimensional Connect-K game engine."""

from hyperzero.game.actions import (
    action_tensor_to_policy,
    logits_to_policy,
    mask_logits,
    normalize_legal_policy,
    policy_to_action_tensor,
    sample_action,
    validate_action,
)
from hyperzero.game.config import GameConfig
from hyperzero.game.env import ConnectKEnv
from hyperzero.game.errors import GameError, InvalidActionError, TerminalStateError
from hyperzero.game.replay import GameReplay
from hyperzero.game.state import GameState, Move
from hyperzero.game.symmetry import Symmetry, gravity_preserving_symmetries

__all__ = [
    "ConnectKEnv",
    "GameConfig",
    "GameError",
    "GameReplay",
    "GameState",
    "InvalidActionError",
    "Move",
    "Symmetry",
    "TerminalStateError",
    "action_tensor_to_policy",
    "gravity_preserving_symmetries",
    "logits_to_policy",
    "mask_logits",
    "normalize_legal_policy",
    "policy_to_action_tensor",
    "sample_action",
    "validate_action",
]

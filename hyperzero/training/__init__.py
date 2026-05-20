"""V1 training utilities for the minimal AlphaZero loop."""

from hyperzero.training.checkpoint import (
    LoadedCheckpoint,
    LoadedUniversalCheckpoint,
    build_checkpoint_agent,
    build_universal_checkpoint_agent,
    build_untrained_agent,
    load_training_checkpoint,
    load_universal_training_checkpoint,
    resolve_device,
)
from hyperzero.training.replay_buffer import ReplayBuffer
from hyperzero.training.self_play import (
    SelfPlayExample,
    SelfPlayGame,
    generate_game,
    generate_games_batched,
)
from hyperzero.training.train import (
    TrainingConfig,
    TrainingMetrics,
    TrainingResult,
    train_v1,
)
from hyperzero.training.train_universal import (
    UniversalTrainingConfig,
    UniversalTrainingMetrics,
    UniversalTrainingResult,
    train_universal,
)
from hyperzero.training.universal_replay import (
    UniversalReplayBuffer,
    UniversalSelfPlayExample,
)
from hyperzero.training.universal_self_play import UniversalGameSpec

__all__ = [
    "ReplayBuffer",
    "SelfPlayExample",
    "SelfPlayGame",
    "TrainingConfig",
    "TrainingMetrics",
    "TrainingResult",
    "UniversalGameSpec",
    "UniversalReplayBuffer",
    "UniversalSelfPlayExample",
    "UniversalTrainingConfig",
    "UniversalTrainingMetrics",
    "UniversalTrainingResult",
    "LoadedCheckpoint",
    "LoadedUniversalCheckpoint",
    "build_checkpoint_agent",
    "build_universal_checkpoint_agent",
    "build_untrained_agent",
    "generate_game",
    "generate_games_batched",
    "load_training_checkpoint",
    "load_universal_training_checkpoint",
    "resolve_device",
    "train_v1",
    "train_universal",
]

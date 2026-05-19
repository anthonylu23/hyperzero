"""Checkpoint loading and agent construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from hyperzero.agents import AlphaZeroAgent
from hyperzero.game.config import GameConfig
from hyperzero.models import NeuralEvaluator, build_policy_value_model


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Loaded model checkpoint and metadata."""

    path: Path
    iteration: int
    game_config: GameConfig
    model: nn.Module
    training_config: dict[str, Any]
    raw: dict[str, Any]


def resolve_device(device: str | torch.device) -> torch.device:
    """Return a torch device, failing early when unavailable."""
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false"
            )
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {resolved.index} is not available")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps is unavailable")
    return resolved


def load_training_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Load a v1 training checkpoint and rebuild its policy-value model."""
    checkpoint_path = Path(path)
    resolved_device = resolve_device(device)
    raw = torch.load(
        checkpoint_path,
        map_location=resolved_device,
        weights_only=False,
    )
    game_config = GameConfig.from_dict(raw["game_config"])
    training_config = dict(raw.get("training_config", {}))
    model = build_policy_value_model(
        game_config,
        model_type=str(training_config.get("model_type", "mlp")),
        hidden_size=int(training_config.get("hidden_size", 64)),
        residual_blocks=int(training_config.get("residual_blocks", 1)),
    )
    model.load_state_dict(raw["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return LoadedCheckpoint(
        path=checkpoint_path,
        iteration=int(raw.get("iteration", 0)),
        game_config=game_config,
        model=model,
        training_config=training_config,
        raw=raw,
    )


def build_checkpoint_agent(
    path: str | Path,
    *,
    simulations: int,
    c_puct: float = 1.5,
    device: str | torch.device = "cpu",
    seed: int | None = None,
    name: str | None = None,
) -> tuple[AlphaZeroAgent, LoadedCheckpoint]:
    """Load a checkpoint and wrap it as an AlphaZeroAgent."""
    checkpoint = load_training_checkpoint(path, device=device)
    agent_name = name or f"checkpoint-{checkpoint.iteration:04d}"
    agent = AlphaZeroAgent(
        NeuralEvaluator(checkpoint.model, device=resolve_device(device)),
        simulations=simulations,
        c_puct=c_puct,
        seed=seed,
        name=agent_name,
    )
    return agent, checkpoint


def build_untrained_agent(
    game_config: GameConfig,
    training_config: dict[str, Any] | None = None,
    *,
    simulations: int,
    c_puct: float = 1.5,
    device: str | torch.device = "cpu",
    seed: int | None = None,
    name: str = "untrained-neural",
) -> AlphaZeroAgent:
    """Build a fresh random-init neural agent with checkpoint-compatible shape."""
    resolved_device = resolve_device(device)
    torch.manual_seed(0 if seed is None else seed)
    training_config = {} if training_config is None else training_config
    model = build_policy_value_model(
        game_config,
        model_type=str(training_config.get("model_type", "mlp")),
        hidden_size=int(training_config.get("hidden_size", 64)),
        residual_blocks=int(training_config.get("residual_blocks", 1)),
    )
    model.to(resolved_device)
    return AlphaZeroAgent(
        NeuralEvaluator(model, device=resolved_device),
        simulations=simulations,
        c_puct=c_puct,
        seed=seed,
        name=name,
    )

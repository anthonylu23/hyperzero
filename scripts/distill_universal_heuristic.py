#!/usr/bin/env python3
"""Fine-tune a universal checkpoint on heuristic policy targets."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyperzero.game import GameConfig, GameState
from hyperzero.training import (
    UniversalSelfPlayExample,
    load_universal_training_checkpoint,
)
from hyperzero.training.train_universal import _training_step
from hyperzero.universal import UniversalEncoderConfig, encode_position


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--examples-output", type=Path, default=None)
    parser.add_argument("--shape", type=int, nargs="+", default=[4, 4, 4, 4])
    parser.add_argument("--connect-k", type=int, default=4)
    parser.add_argument("--gravity-axis", type=int, default=0)
    parser.add_argument("--config-id", default="4d_4x4x4x4_k4")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--max-plies-per-game", type=int, default=96)
    parser.add_argument("--max-examples", type=int, default=20_000)
    parser.add_argument("--progress-interval", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--behavior-epsilon", type=float, default=0.25)
    parser.add_argument("--opponent-scale", type=float, default=1.25)
    parser.add_argument("--center-scale", type=float, default=0.05)
    parser.add_argument("--value-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.behavior_epsilon <= 1.0:
        raise ValueError("behavior-epsilon must be in [0, 1]")
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    loaded = load_universal_training_checkpoint(args.checkpoint, device=device)
    model = loaded.model
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    config = GameConfig(
        shape=tuple(args.shape),
        connect_k=args.connect_k,
        gravity_axis=args.gravity_axis,
    )
    examples = _generate_heuristic_examples(
        config,
        config_id=args.config_id,
        games=args.games,
        max_plies_per_game=args.max_plies_per_game,
        max_examples=args.max_examples,
        progress_interval=args.progress_interval,
        behavior_epsilon=args.behavior_epsilon,
        opponent_scale=args.opponent_scale,
        center_scale=args.center_scale,
        rng=rng,
    )
    if args.examples_output is not None:
        _save_teacher_replay(
            args.examples_output,
            examples,
            metadata={
                "source_checkpoint": str(args.checkpoint),
                "config_id": args.config_id,
                "shape": list(config.shape),
                "connect_k": config.connect_k,
                "gravity_axis": config.gravity_axis,
                "games": args.games,
                "max_plies_per_game": args.max_plies_per_game,
                "max_examples": args.max_examples,
                "examples": len(examples),
                "behavior_epsilon": args.behavior_epsilon,
                "opponent_scale": args.opponent_scale,
                "center_scale": args.center_scale,
                "seed": args.seed,
            },
        )
    if len(examples) < args.batch_size:
        raise ValueError(
            f"generated {len(examples)} examples, fewer than batch size "
            f"{args.batch_size}"
        )

    losses = []
    for step in range(1, args.steps + 1):
        batch_examples = random.sample(examples, args.batch_size)
        losses.append(
            _training_step(
                model,
                optimizer,
                batch_examples,
                value_weight=args.value_weight,
                model_config=loaded.model_config,
                device=device,
            )
        )
        if step == 1 or step % 100 == 0 or step == args.steps:
            recent = losses[-min(100, len(losses)) :]
            summary = _loss_summary(recent)
            print(
                f"step={step} policy_loss={summary['policy_loss']:.4f} "
                f"value_loss={summary['value_loss']:.4f} "
                f"total_loss={summary['total_loss']:.4f}",
                flush=True,
            )

    metrics = {
        "source_checkpoint": str(args.checkpoint),
        "source_iteration": loaded.iteration,
        "output_checkpoint": str(args.output_checkpoint),
        "config_id": args.config_id,
        "games": args.games,
        "max_plies_per_game": args.max_plies_per_game,
        "max_examples": args.max_examples,
        "examples": len(examples),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "behavior_epsilon": args.behavior_epsilon,
        "opponent_scale": args.opponent_scale,
        "center_scale": args.center_scale,
        "value_weight": args.value_weight,
        "loss": _loss_summary(losses),
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _save_checkpoint(args.output_checkpoint, loaded, model, metrics)


def _generate_heuristic_examples(
    config: GameConfig,
    *,
    config_id: str,
    games: int,
    max_plies_per_game: int,
    max_examples: int,
    progress_interval: int,
    behavior_epsilon: float,
    opponent_scale: float,
    center_scale: float,
    rng: np.random.Generator,
) -> list[UniversalSelfPlayExample]:
    examples: list[UniversalSelfPlayExample] = []
    encoder = UniversalEncoderConfig(line_features=True)
    for game_index in range(1, games + 1):
        state = GameState.new(config, use_line_counts=True)
        while not state.terminal and state.ply < max_plies_per_game:
            legal_mask = state.legal_mask()
            target_action = _fast_line_teacher_action(
                state,
                encoder=encoder,
                opponent_scale=opponent_scale,
                center_scale=center_scale,
                rng=rng,
            )
            policy = np.zeros(config.num_actions, dtype=np.float32)
            policy[target_action] = 1.0
            examples.append(
                UniversalSelfPlayExample(
                    config_id=config_id,
                    game_config=config,
                    board=state.canonical_board(flat=True).astype(np.float32),
                    policy=policy,
                    value=0.0,
                    legal_mask=legal_mask.astype(bool),
                    player_to_move=state.player_to_move,
                    ply=state.ply,
                )
            )
            if rng.random() < behavior_epsilon:
                action = int(rng.choice(np.flatnonzero(legal_mask)))
            else:
                action = target_action
            state.make_move(action)
            if len(examples) >= max_examples:
                break
        if progress_interval > 0 and (
            game_index == 1 or game_index % progress_interval == 0
        ):
            print(
                f"generated_games={game_index} examples={len(examples)}",
                flush=True,
            )
        if len(examples) >= max_examples:
            break
    return examples


def _fast_line_teacher_action(
    state: GameState,
    *,
    encoder: UniversalEncoderConfig,
    opponent_scale: float,
    center_scale: float,
    rng: np.random.Generator,
) -> int:
    legal_mask = state.legal_mask()
    position = encode_position(
        state.config,
        board=state.canonical_board(flat=True),
        legal_mask=legal_mask,
        ply=state.ply,
        encoder_config=encoder,
        line_counts=_canonical_line_counts(state),
    )
    line_offset = UniversalEncoderConfig().feature_size
    line_features = position.action_features[:, line_offset : line_offset + 4]
    scores = (
        line_features[:, 0]
        + opponent_scale * line_features[:, 1]
        + 25.0 * line_features[:, 2]
        + 20.0 * line_features[:, 3]
        + _center_scores(state.config, center_scale)
    )
    scores = np.where(legal_mask, scores, -np.inf)
    best = np.flatnonzero(scores == np.max(scores))
    return int(rng.choice(best))


def _canonical_line_counts(state: GameState) -> tuple[np.ndarray, np.ndarray] | None:
    if state.line_counts is None:
        return None
    own_index = 1 if state.player_to_move == 1 else 0
    opponent_index = 1 - own_index
    return (
        state.line_counts[own_index].astype(np.int16, copy=False),
        state.line_counts[opponent_index].astype(np.int16, copy=False),
    )


def _center_scores(config: GameConfig, center_scale: float) -> np.ndarray:
    if center_scale == 0.0 or not config.action_shape:
        return np.zeros(config.num_actions, dtype=np.float32)
    center = (np.asarray(config.action_shape, dtype=np.float64) - 1.0) / 2.0
    max_distance = float(np.linalg.norm(center))
    if max_distance == 0.0:
        return np.full(config.num_actions, center_scale, dtype=np.float32)
    scores = []
    for action in range(config.num_actions):
        coord = np.asarray(config.column_coord(action), dtype=np.float64)
        distance = float(np.linalg.norm(coord - center))
        scores.append(center_scale * (1.0 - distance / max_distance))
    return np.asarray(scores, dtype=np.float32)


def _loss_summary(losses: list[tuple[float, float, float]]) -> dict[str, float]:
    arr = np.asarray(losses, dtype=np.float64)
    return {
        "policy_loss": float(arr[:, 0].mean()),
        "value_loss": float(arr[:, 1].mean()),
        "total_loss": float(arr[:, 2].mean()),
    }


def _save_checkpoint(
    path: Path,
    loaded,
    model: torch.nn.Module,
    metrics: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = dict(loaded.raw)
    raw["model_state_dict"] = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    raw["distillation_metrics"] = metrics
    raw["checkpoint_note"] = "heuristic policy distillation"
    torch.save(raw, path)


def _save_teacher_replay(
    path: Path,
    examples: list[UniversalSelfPlayExample],
    *,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "universal_teacher_replay_v1",
            "metadata": metadata,
            "examples": examples,
        },
        path,
    )


if __name__ == "__main__":
    main()

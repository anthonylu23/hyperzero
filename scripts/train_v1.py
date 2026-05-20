#!/usr/bin/env python3
"""Run a small v1 AlphaZero training smoke loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyperzero.game import GameConfig
from hyperzero.training import TrainingConfig, train_v1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs="+", default=[3, 3])
    parser.add_argument("--connect-k", type=int, default=3)
    parser.add_argument("--gravity-axis", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--training-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=2_000)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=1)
    parser.add_argument(
        "--model-type",
        choices=("mlp", "line_mlp", "cnn", "resnet", "line_resnet", "transformer"),
        default="mlp",
    )
    parser.add_argument(
        "--symmetry-augmentation",
        choices=("none", "random"),
        default="none",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-keep-last", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--eval-games", type=int, default=0)
    parser.add_argument(
        "--eval-opponents",
        nargs="+",
        choices=("random", "tactical", "heuristic", "mcts"),
        default=[],
    )
    parser.add_argument("--eval-simulations", type=int, default=8)
    parser.add_argument("--eval-mcts-simulations", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument(
        "--eval-score-weights",
        type=json.loads,
        default=None,
        help=(
            "JSON object mapping eval opponent names to score weights for "
            "best checkpoint selection."
        ),
    )
    parser.add_argument("--batched-self-play", action="store_true")
    parser.add_argument("--max-active-self-play-games", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    game_config = GameConfig(
        shape=tuple(args.shape),
        connect_k=args.connect_k,
        gravity_axis=args.gravity_axis,
    )
    result = train_v1(
        TrainingConfig(
            game_config=game_config,
            iterations=args.iterations,
            self_play_games_per_iteration=args.games,
            puct_simulations=args.simulations,
            replay_capacity=args.replay_capacity,
            batch_size=args.batch_size,
            training_steps_per_iteration=args.training_steps,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            residual_blocks=args.residual_blocks,
            model_type=args.model_type,
            symmetry_augmentation=args.symmetry_augmentation,
            seed=args.seed,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_keep_last=args.checkpoint_keep_last,
            resume_from_checkpoint=args.resume_from_checkpoint,
            metrics_path=args.metrics_path,
            eval_games_per_iteration=args.eval_games,
            eval_opponents=tuple(args.eval_opponents),
            eval_simulations=args.eval_simulations,
            eval_mcts_simulations=args.eval_mcts_simulations,
            eval_interval=args.eval_interval,
            eval_score_weights=args.eval_score_weights,
            batched_self_play=args.batched_self_play,
            max_active_self_play_games=args.max_active_self_play_games,
        )
    )
    for metric in result.metrics:
        eval_summary = " ".join(
            f"eval_{name}_win_rate={stats['agent_a_win_rate']:.3f}"
            for name, stats in metric.evaluations.items()
        )
        print(
            f"iteration={metric.iteration} games={metric.self_play_games} "
            f"examples={metric.self_play_examples} replay={metric.replay_size} "
            f"model={metric.model_type} aug={metric.symmetry_augmentation} "
            f"policy_loss={metric.policy_loss:.4f} "
            f"value_loss={metric.value_loss:.4f} total_loss={metric.total_loss:.4f} "
            f"eval_score={metric.eval_score} best={metric.is_best_checkpoint} "
            f"avg_len={metric.average_game_length:.2f} "
            f"batched_self_play={metric.batched_self_play} "
            f"iter_time={metric.iteration_time_seconds:.2f}s "
            f"self_play_time={metric.self_play_time_seconds:.2f}s "
            f"train_step_time={metric.training_step_time_seconds:.2f}s "
            f"eval_time={metric.eval_time_seconds:.2f}s "
            f"inference_time={metric.total_inference_time_seconds:.2f}s "
            f"checkpoint={metric.checkpoint_path} {eval_summary}".rstrip()
        )


if __name__ == "__main__":
    main()

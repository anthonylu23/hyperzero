#!/usr/bin/env python3
"""Run universal mixed-variant AlphaZero-style training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyperzero.game import GameConfig
from hyperzero.training import (
    UniversalGameSpec,
    UniversalTrainingConfig,
    train_universal,
)

DEFAULT_VARIANTS = [
    {
        "config_id": "2d_4x4_k3",
        "shape": [4, 4],
        "connect_k": 3,
        "gravity_axis": 0,
        "games": 1,
    },
    {
        "config_id": "3d_4x4x4_k4",
        "shape": [4, 4, 4],
        "connect_k": 4,
        "gravity_axis": 0,
        "games": 1,
    },
    {
        "config_id": "4d_4x4x4x4_k4",
        "shape": [4, 4, 4, 4],
        "connect_k": 4,
        "gravity_axis": 0,
        "games": 1,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants-json",
        type=Path,
        default=None,
        help=(
            "JSON file containing variant objects with config_id, shape, "
            "connect_k, gravity_axis, and games."
        ),
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--training-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-rank", type=int, default=4)
    parser.add_argument("--max-board-extent", type=int, default=8)
    parser.add_argument("--line-features", action="store_true")
    parser.add_argument("--input-layer-norm", action="store_true")
    parser.add_argument("--rank-adapters", action="store_true")
    parser.add_argument("--rank-head-adapters", action="store_true")
    parser.add_argument("--adapter-size", type=int, default=0)
    parser.add_argument("--line-policy-residual", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.0)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.0)
    parser.add_argument("--root-temperature", type=float, default=0.0)
    parser.add_argument("--root-temperature-move-cutoff", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-keep-last", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--reset-optimizer-on-resume", action="store_true")
    parser.add_argument("--reset-replay-on-resume", action="store_true")
    parser.add_argument("--reset-rng-on-resume", action="store_true")
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--eval-games", type=int, default=0)
    parser.add_argument(
        "--eval-opponents",
        nargs="+",
        choices=("random", "tactical", "heuristic", "mcts"),
        default=[],
    )
    parser.add_argument("--eval-simulations", type=int, default=4)
    parser.add_argument("--eval-mcts-simulations", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument(
        "--eval-score-weights",
        type=json.loads,
        default=None,
        help="JSON object mapping eval opponent names to score weights.",
    )
    parser.add_argument(
        "--eval-score-floors",
        type=json.loads,
        default=None,
        help=(
            "JSON object with default or per-variant minimum opponent win rates "
            "required before a checkpoint can become best."
        ),
    )
    parser.add_argument("--batched-self-play", action="store_true")
    parser.add_argument("--max-active-self-play-games", type=int, default=None)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--self-play-workers", type=int, default=1)
    parser.add_argument("--central-batched-self-play", action="store_true")
    parser.add_argument("--unbalanced-replay", action="store_true")
    parser.add_argument("--teacher-replay-path", type=Path, default=None)
    parser.add_argument("--teacher-batch-fraction", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_universal(
        UniversalTrainingConfig(
            game_specs=_load_game_specs(args.variants_json),
            iterations=args.iterations,
            puct_simulations=args.simulations,
            replay_capacity=args.replay_capacity,
            batch_size=args.batch_size,
            training_steps_per_iteration=args.training_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            value_weight=args.value_weight,
            c_puct=args.c_puct,
            hidden_size=args.hidden_size,
            residual_blocks=args.residual_blocks,
            heads=args.heads,
            max_rank=args.max_rank,
            max_board_extent=args.max_board_extent,
            line_features=args.line_features,
            input_layer_norm=args.input_layer_norm,
            rank_adapters=args.rank_adapters,
            rank_head_adapters=args.rank_head_adapters,
            adapter_size=args.adapter_size,
            line_policy_residual=args.line_policy_residual,
            seed=args.seed,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_keep_last=args.checkpoint_keep_last,
            resume_from_checkpoint=args.resume_from_checkpoint,
            reset_optimizer_on_resume=args.reset_optimizer_on_resume,
            reset_replay_on_resume=args.reset_replay_on_resume,
            reset_rng_on_resume=args.reset_rng_on_resume,
            metrics_path=args.metrics_path,
            eval_games_per_variant=args.eval_games,
            eval_opponents=tuple(args.eval_opponents),
            eval_simulations=args.eval_simulations,
            eval_mcts_simulations=args.eval_mcts_simulations,
            eval_interval=args.eval_interval,
            eval_score_weights=args.eval_score_weights,
            eval_score_floors=args.eval_score_floors,
            batched_self_play=args.batched_self_play,
            max_active_self_play_games=args.max_active_self_play_games,
            eval_workers=args.eval_workers,
            self_play_workers=args.self_play_workers,
            central_batched_self_play=args.central_batched_self_play,
            balanced_replay=not args.unbalanced_replay,
            root_dirichlet_alpha=args.root_dirichlet_alpha,
            root_exploration_fraction=args.root_exploration_fraction,
            root_temperature=args.root_temperature,
            root_temperature_move_cutoff=args.root_temperature_move_cutoff,
            teacher_replay_path=args.teacher_replay_path,
            teacher_batch_fraction=args.teacher_batch_fraction,
        )
    )
    for metric in result.metrics:
        eval_summary = " ".join(
            f"{variant}_{opponent}_win_rate={stats['agent_a_win_rate']:.3f}"
            for variant, variant_stats in metric.evaluations.items()
            for opponent, stats in variant_stats.items()
        )
        print(
            f"iteration={metric.iteration} games={metric.self_play_games} "
            f"examples={metric.self_play_examples} replay={metric.replay_size} "
            f"policy_loss={metric.policy_loss:.4f} "
            f"value_loss={metric.value_loss:.4f} total_loss={metric.total_loss:.4f} "
            f"eval_score={metric.eval_score} best={metric.is_best_checkpoint} "
            f"floor_passed={metric.eval_floor_passed} "
            f"min_tactical={metric.min_tactical_win_rate} "
            f"iter_time={metric.iteration_time_seconds:.2f}s "
            f"self_play_time={metric.self_play_time_seconds:.2f}s "
            f"train_step_time={metric.training_step_time_seconds:.2f}s "
            f"eval_time={metric.eval_time_seconds:.2f}s "
            f"self_play_workers={metric.self_play_workers} "
            f"eval_workers={metric.eval_workers} "
            f"central_batched={metric.central_batched_self_play} "
            f"self_play_infer_states={metric.self_play_inference_states} "
            f"self_play_infer_batches={metric.self_play_inference_batches} "
            f"checkpoint={metric.checkpoint_path} {eval_summary}".rstrip()
        )


def _load_game_specs(path: Path | None) -> tuple[UniversalGameSpec, ...]:
    rows = DEFAULT_VARIANTS if path is None else json.loads(path.read_text())
    specs = []
    for row in rows:
        specs.append(
            UniversalGameSpec(
                config_id=str(row["config_id"]),
                game_config=GameConfig(
                    shape=tuple(int(size) for size in row["shape"]),
                    connect_k=int(row["connect_k"]),
                    gravity_axis=int(row.get("gravity_axis", 0)),
                ),
                self_play_games_per_iteration=int(row.get("games", 1)),
            )
        )
    return tuple(specs)


if __name__ == "__main__":
    main()

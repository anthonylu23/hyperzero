#!/usr/bin/env python3
"""Robustly evaluate universal checkpoints across all trained variants."""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hyperzero.agents import HeuristicAgent, MCTSAgent, RandomAgent, TacticalAgent
from hyperzero.eval import evaluate_matchup
from hyperzero.game import GameConfig
from hyperzero.training import UniversalGameSpec, build_universal_checkpoint_agent

DEFAULT_SCORE_WEIGHTS = {"heuristic": 0.55, "tactical": 0.35, "random": 0.10}
DEFAULT_SCORE_FLOORS = {
    "default": {"random": 0.85, "tactical": 0.50},
    "2d_6x7_k4": {"heuristic": 0.125},
    "4d_4x4x4x4_k4": {"heuristic": 0.375},
}
BASELINE_NAMES = ("random", "tactical", "heuristic", "mcts")


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    label: str
    path: Path


@dataclass(frozen=True, slots=True)
class EvalTask:
    checkpoint: CheckpointSpec
    variant: UniversalGameSpec
    opponent: str
    seed_index: int
    games: int
    simulations: int
    mcts_simulations: int
    c_puct: float
    base_seed: int
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Checkpoint to evaluate. May be passed multiple times.",
    )
    parser.add_argument(
        "--checkpoints-json",
        type=Path,
        help="Optional JSON list of {label, path} checkpoint objects.",
    )
    parser.add_argument(
        "--variants-json",
        type=Path,
        help="Optional variant JSON. Defaults to the first checkpoint's game specs.",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=BASELINE_NAMES,
        default=["random", "tactical", "heuristic"],
    )
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=24)
    parser.add_argument("--mcts-simulations", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--eval-score-weights",
        type=json.loads,
        default=DEFAULT_SCORE_WEIGHTS,
        help="JSON object mapping opponent names to score weights.",
    )
    parser.add_argument(
        "--eval-score-floors",
        type=json.loads,
        default=DEFAULT_SCORE_FLOORS,
        help="JSON object with default or per-variant opponent floors.",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--jsonl-output", type=Path)
    return parser.parse_args()


def parse_checkpoint_spec(value: str) -> CheckpointSpec:
    if "=" in value:
        label, path = value.split("=", 1)
        if not label:
            raise ValueError("checkpoint label cannot be empty")
    else:
        path = value
        label = Path(path).stem
    checkpoint_path = Path(path)
    return CheckpointSpec(label=label, path=checkpoint_path)


def load_checkpoint_specs(args: argparse.Namespace) -> list[CheckpointSpec]:
    specs = [parse_checkpoint_spec(value) for value in args.checkpoint]
    if args.checkpoints_json is not None:
        rows = json.loads(args.checkpoints_json.read_text(encoding="utf-8"))
        for row in rows:
            specs.append(
                CheckpointSpec(label=str(row["label"]), path=Path(row["path"]))
            )
    if not specs:
        raise SystemExit("at least one --checkpoint or --checkpoints-json is required")
    for spec in specs:
        if not spec.path.exists():
            raise SystemExit(f"checkpoint does not exist: {spec.path}")
    return specs


def load_variants(
    path: Path | None,
    checkpoint: CheckpointSpec,
) -> tuple[UniversalGameSpec, ...]:
    if path is None:
        _, loaded = build_universal_checkpoint_agent(
            checkpoint.path,
            simulations=1,
            device="cpu",
        )
        return loaded.game_specs
    rows = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        UniversalGameSpec(
            config_id=str(row["config_id"]),
            game_config=GameConfig(
                shape=tuple(int(size) for size in row["shape"]),
                connect_k=int(row["connect_k"]),
                gravity_axis=int(row.get("gravity_axis", 0)),
            ),
            self_play_games_per_iteration=int(row.get("games", 1)),
        )
        for row in rows
    )


def build_baseline(name: str, *, seed: int, mcts_simulations: int):
    if name == "random":
        return RandomAgent(seed=seed, name=f"random-{seed}")
    if name == "tactical":
        return TacticalAgent(seed=seed, name=f"tactical-{seed}")
    if name == "heuristic":
        return HeuristicAgent(seed=seed, name=f"heuristic-{seed}")
    if name == "mcts":
        return MCTSAgent(
            simulations=mcts_simulations,
            seed=seed,
            name=f"mcts-{mcts_simulations}-{seed}",
        )
    raise ValueError(f"unknown baseline: {name}")


def run_eval_task(task: EvalTask) -> dict[str, Any]:
    torch.set_num_threads(1)
    offset = task.seed_index * 100_000
    agent_seed = task.base_seed + offset + 17
    opponent_seed = task.base_seed + offset + 29
    agent, checkpoint = build_universal_checkpoint_agent(
        task.checkpoint.path,
        simulations=task.simulations,
        c_puct=task.c_puct,
        device=task.device,
        seed=agent_seed,
        name=f"{task.checkpoint.label}-{task.variant.config_id}",
    )
    opponent = build_baseline(
        task.opponent,
        seed=opponent_seed,
        mcts_simulations=task.mcts_simulations,
    )
    stats = evaluate_matchup(
        task.variant.game_config,
        agent,
        opponent,
        games=task.games,
    )
    return {
        "checkpoint": task.checkpoint.label,
        "checkpoint_path": str(task.checkpoint.path),
        "iteration": checkpoint.iteration,
        "variant": task.variant.config_id,
        "opponent": task.opponent,
        "seed_index": task.seed_index,
        "agent_seed": agent_seed,
        "opponent_seed": opponent_seed,
        "agent_simulations": task.simulations,
        "mcts_simulations": task.mcts_simulations,
        "stats": stats.to_dict(),
    }


def aggregate_records(
    records: list[dict[str, Any]],
    variants: tuple[UniversalGameSpec, ...],
    weights: dict[str, float],
    floors: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    labels = sorted({str(record["checkpoint"]) for record in records})
    summaries = []
    for label in labels:
        label_records = [
            record for record in records if str(record["checkpoint"]) == label
        ]
        evaluations: dict[str, dict[str, dict[str, float | int]]] = {}
        for variant in variants:
            variant_records = [
                record
                for record in label_records
                if record["variant"] == variant.config_id
            ]
            opponent_stats = {}
            for opponent in sorted({record["opponent"] for record in variant_records}):
                opponent_records = [
                    record
                    for record in variant_records
                    if record["opponent"] == opponent
                ]
                games = sum(
                    int(record["stats"]["games"]) for record in opponent_records
                )
                wins = sum(
                    int(record["stats"]["agent_a_wins"])
                    for record in opponent_records
                )
                losses = sum(
                    int(record["stats"]["agent_b_wins"])
                    for record in opponent_records
                )
                draws = sum(
                    int(record["stats"]["draws"]) for record in opponent_records
                )
                win_rate = wins / games if games else 0.0
                opponent_stats[opponent] = {
                    "games": games,
                    "agent_a_wins": wins,
                    "agent_b_wins": losses,
                    "draws": draws,
                    "agent_a_win_rate": win_rate,
                    "agent_b_win_rate": losses / games if games else 0.0,
                    "draw_rate": draws / games if games else 0.0,
                    "win_rate_ci95": _normal_ci95(win_rate, games),
                }
            evaluations[variant.config_id] = opponent_stats
        score = score_evaluations(variants, evaluations, weights)
        failures = eval_floor_failures(variants, evaluations, floors)
        summaries.append(
            {
                "checkpoint": label,
                "checkpoint_path": label_records[0]["checkpoint_path"],
                "iteration": label_records[0]["iteration"],
                "eval_score": score,
                "floor_passed": not failures,
                "floor_failures": failures,
                "evaluations": evaluations,
            }
        )
    return sorted(
        summaries,
        key=lambda row: -1.0 if row["eval_score"] is None else -row["eval_score"],
    )


def _normal_ci95(win_rate: float, games: int) -> tuple[float, float]:
    if games <= 0:
        return (0.0, 0.0)
    half_width = 1.96 * math.sqrt(win_rate * (1.0 - win_rate) / games)
    return (max(0.0, win_rate - half_width), min(1.0, win_rate + half_width))


def score_evaluations(
    variants: tuple[UniversalGameSpec, ...],
    evaluations: dict[str, dict[str, dict[str, float | int]]],
    weights: dict[str, float],
) -> float | None:
    per_variant_scores = []
    for variant in variants:
        variant_results = evaluations.get(variant.config_id, {})
        total_weight = 0.0
        score = 0.0
        for opponent, weight in weights.items():
            stats = variant_results.get(opponent)
            if stats is None:
                continue
            score += float(weight) * float(stats["agent_a_win_rate"])
            total_weight += float(weight)
        if total_weight > 0.0:
            per_variant_scores.append(score / total_weight)
    if not per_variant_scores:
        return None
    return 0.5 * (
        sum(per_variant_scores) / len(per_variant_scores)
    ) + 0.5 * min(per_variant_scores)


def eval_floor_failures(
    variants: tuple[UniversalGameSpec, ...],
    evaluations: dict[str, dict[str, dict[str, float | int]]],
    floors: dict[str, Any] | None,
) -> list[str]:
    if not floors:
        return []
    failures = []
    for variant in variants:
        thresholds = floor_thresholds_for_variant(floors, variant.config_id)
        variant_results = evaluations.get(variant.config_id, {})
        for opponent, threshold in thresholds.items():
            stats = variant_results.get(opponent)
            if stats is None:
                failures.append(f"{variant.config_id}:{opponent}=missing<{threshold:.3f}")
                continue
            win_rate = float(stats["agent_a_win_rate"])
            if win_rate < threshold:
                failures.append(
                    f"{variant.config_id}:{opponent}={win_rate:.3f}<{threshold:.3f}"
                )
    return failures


def floor_thresholds_for_variant(
    floors: dict[str, Any],
    config_id: str,
) -> dict[str, float]:
    thresholds = {}
    for key, value in floors.items():
        if isinstance(value, dict):
            if key == "default" or key == config_id:
                thresholds.update(
                    {opponent: float(score) for opponent, score in value.items()}
                )
        elif key in BASELINE_NAMES:
            thresholds[key] = float(value)
    return thresholds


def print_summary(summaries: list[dict[str, Any]]) -> None:
    print("checkpoint\titeration\teval_score\tfloor_passed\tfloor_failures")
    for row in summaries:
        score = row["eval_score"]
        score_text = "-" if score is None else f"{score:.4f}"
        failures = "; ".join(row["floor_failures"])
        print(
            f"{row['checkpoint']}\t{row['iteration']}\t{score_text}\t"
            f"{row['floor_passed']}\t{failures}"
        )
    for row in summaries:
        print(f"\n[{row['checkpoint']}]")
        print("variant\topponent\twin_rate\tgames\tci95")
        for variant, opponent_rows in row["evaluations"].items():
            for opponent, stats in opponent_rows.items():
                lo, hi = stats["win_rate_ci95"]
                print(
                    f"{variant}\t{opponent}\t"
                    f"{float(stats['agent_a_win_rate']):.3f}\t"
                    f"{int(stats['games'])}\t[{lo:.3f},{hi:.3f}]"
                )


def main() -> None:
    args = parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be positive")
    if args.seed_count <= 0:
        raise SystemExit("--seed-count must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    checkpoints = load_checkpoint_specs(args)
    variants = load_variants(args.variants_json, checkpoints[0])
    tasks = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        for seed_index in range(args.seed_count):
            for variant_index, variant in enumerate(variants):
                for opponent_index, opponent in enumerate(args.opponents):
                    base_seed = (
                        args.seed
                        + checkpoint_index * 1_000_000
                        + seed_index * 100_000
                        + variant_index * 1_000
                        + opponent_index * 100
                    )
                    tasks.append(
                        EvalTask(
                            checkpoint=checkpoint,
                            variant=variant,
                            opponent=opponent,
                            seed_index=seed_index,
                            games=args.games,
                            simulations=args.simulations,
                            mcts_simulations=args.mcts_simulations,
                            c_puct=args.c_puct,
                            base_seed=base_seed,
                            device=args.device,
                        )
                    )

    records: list[dict[str, Any]] = []
    output_handle = None
    if args.jsonl_output is not None:
        args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.jsonl_output.open("w", encoding="utf-8")
    try:
        if args.workers == 1:
            for task in tasks:
                record = run_eval_task(task)
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
                if output_handle is not None:
                    output_handle.write(json.dumps(record, sort_keys=True) + "\n")
                    output_handle.flush()
        else:
            ctx = get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=ctx,
            ) as executor:
                future_to_task = {
                    executor.submit(run_eval_task, task): task for task in tasks
                }
                for future in as_completed(future_to_task):
                    record = future.result()
                    records.append(record)
                    print(json.dumps(record, sort_keys=True), flush=True)
                    if output_handle is not None:
                        output_handle.write(json.dumps(record, sort_keys=True) + "\n")
                        output_handle.flush()
    finally:
        if output_handle is not None:
            output_handle.close()

    summaries = aggregate_records(
        records,
        variants,
        args.eval_score_weights,
        args.eval_score_floors,
    )
    payload = {
        "checkpoints": [
            {"label": checkpoint.label, "path": str(checkpoint.path)}
            for checkpoint in checkpoints
        ],
        "variants": [variant.to_dict() for variant in variants],
        "opponents": args.opponents,
        "games": args.games,
        "seed_count": args.seed_count,
        "simulations": args.simulations,
        "mcts_simulations": args.mcts_simulations,
        "eval_score_weights": args.eval_score_weights,
        "eval_score_floors": args.eval_score_floors,
        "summaries": summaries,
        "records": records,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print_summary(summaries)


if __name__ == "__main__":
    main()

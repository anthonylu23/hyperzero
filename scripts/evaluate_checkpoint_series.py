#!/usr/bin/env python3
"""Evaluate every v1 checkpoint in a directory against fixed opponents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyperzero.agents import HeuristicAgent, MCTSAgent, RandomAgent, TacticalAgent
from hyperzero.eval import evaluate_matchup
from hyperzero.training import build_checkpoint_agent


def build_baseline(name: str, *, seed: int, mcts_simulations: int):
    """Build a non-neural baseline agent."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=("random", "tactical", "heuristic", "mcts"),
        default=["random", "tactical", "heuristic"],
    )
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--mcts-simulations", type=int, default=100)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--promotion-opponent", default="heuristic")
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    parser.add_argument("--checkpoint-stride", type=int, default=1)
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--best-only", action="store_true")
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--jsonl-output", type=Path)
    return parser.parse_args()


def select_checkpoints(
    checkpoint_dir: Path,
    *,
    checkpoint_stride: int = 1,
    latest_only: bool = False,
    best_only: bool = False,
    max_checkpoints: int | None = None,
) -> list[Path]:
    """Select checkpoints for series eval with optional downsampling."""
    if checkpoint_stride <= 0:
        raise ValueError("checkpoint_stride must be positive")
    if max_checkpoints is not None and max_checkpoints <= 0:
        raise ValueError("max_checkpoints must be positive when set")
    if latest_only and best_only:
        raise ValueError("latest_only and best_only are mutually exclusive")

    checkpoints = sorted(checkpoint_dir.glob("iteration_*.pt"))
    if best_only:
        best_checkpoint = checkpoint_dir / "best_by_eval_score.pt"
        return [best_checkpoint] if best_checkpoint.exists() else []
    if latest_only:
        return checkpoints[-1:] if checkpoints else []

    selected = checkpoints[::checkpoint_stride]
    if checkpoints and checkpoints[-1] not in selected:
        selected.append(checkpoints[-1])
    if max_checkpoints is not None:
        selected = selected[-max_checkpoints:]
    return selected


def main() -> None:
    args = parse_args()
    checkpoints = select_checkpoints(
        args.checkpoint_dir,
        checkpoint_stride=args.checkpoint_stride,
        latest_only=args.latest_only,
        best_only=args.best_only,
        max_checkpoints=args.max_checkpoints,
    )
    if not checkpoints:
        raise SystemExit(
            f"no selected checkpoints found in {args.checkpoint_dir}"
        )

    output_handle = None
    if args.jsonl_output is not None:
        args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.jsonl_output.open("w", encoding="utf-8")
    for checkpoint_index, checkpoint_path in enumerate(checkpoints):
        agent, checkpoint = build_checkpoint_agent(
            checkpoint_path,
            simulations=args.simulations,
            c_puct=args.c_puct,
            device=args.device,
            seed=args.seed + checkpoint_index,
            name=f"checkpoint-{checkpoint_path.stem}",
        )
        for opponent_index, opponent_name in enumerate(args.opponents):
            opponent = build_baseline(
                opponent_name,
                seed=args.seed + 10_000 + checkpoint_index * 100 + opponent_index,
                mcts_simulations=args.mcts_simulations,
            )
            stats = evaluate_matchup(
                checkpoint.game_config,
                agent,
                opponent,
                games=args.games,
            )
            passed = (
                opponent_name == args.promotion_opponent
                and stats.agent_a_win_rate > args.promotion_threshold
            )
            payload = {
                "checkpoint": str(checkpoint_path),
                "iteration": checkpoint.iteration,
                "opponent": opponent_name,
                "agent_simulations": args.simulations,
                "mcts_simulations": args.mcts_simulations,
                "promotion_threshold": args.promotion_threshold,
                "promotion_passed": passed,
                "stats": stats.to_dict(),
            }
            line = json.dumps(payload, sort_keys=True)
            print(line)
            if output_handle is not None:
                output_handle.write(line + "\n")
                output_handle.flush()

    if output_handle is not None:
        output_handle.close()


if __name__ == "__main__":
    main()

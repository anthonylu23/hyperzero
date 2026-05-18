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
    parser.add_argument("--jsonl-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = sorted(args.checkpoint_dir.glob("iteration_*.pt"))
    if not checkpoints:
        raise SystemExit(
            f"no iteration_*.pt checkpoints found in {args.checkpoint_dir}"
        )

    lines: list[str] = []
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
            lines.append(line)
            print(line)

    if args.jsonl_output is not None:
        args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a v1 neural checkpoint against a baseline or another checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyperzero.agents import HeuristicAgent, MCTSAgent, RandomAgent, TacticalAgent
from hyperzero.eval import evaluate_matchup
from hyperzero.training import (
    build_checkpoint_agent,
    build_untrained_agent,
)


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--opponent",
        choices=("random", "tactical", "heuristic", "mcts", "checkpoint", "untrained"),
        default="heuristic",
    )
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--opponent-simulations", type=int, default=None)
    parser.add_argument("--mcts-simulations", type=int, default=100)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-swap-sides", action="store_true")
    parser.add_argument("--promotion-threshold", type=float, default=None)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opponent_simulations = (
        args.simulations
        if args.opponent_simulations is None
        else args.opponent_simulations
    )
    agent, checkpoint = build_checkpoint_agent(
        args.checkpoint,
        simulations=args.simulations,
        c_puct=args.c_puct,
        device=args.device,
        seed=args.seed,
        name=f"checkpoint-{args.checkpoint.stem}",
    )

    if args.opponent == "checkpoint":
        if args.opponent_checkpoint is None:
            raise SystemExit(
                "--opponent-checkpoint is required for checkpoint opponent"
            )
        opponent, opponent_checkpoint = build_checkpoint_agent(
            args.opponent_checkpoint,
            simulations=opponent_simulations,
            c_puct=args.c_puct,
            device=args.device,
            seed=args.seed + 1,
            name=f"checkpoint-{args.opponent_checkpoint.stem}",
        )
        opponent_metadata = {
            "type": "checkpoint",
            "path": str(args.opponent_checkpoint),
            "iteration": opponent_checkpoint.iteration,
        }
    elif args.opponent == "untrained":
        opponent = build_untrained_agent(
            checkpoint.game_config,
            checkpoint.training_config,
            simulations=opponent_simulations,
            c_puct=args.c_puct,
            device=args.device,
            seed=args.seed + 1,
        )
        opponent_metadata = {"type": "untrained"}
    else:
        opponent = build_baseline(
            args.opponent,
            seed=args.seed + 1,
            mcts_simulations=args.mcts_simulations,
        )
        opponent_metadata = {"type": args.opponent}

    stats = evaluate_matchup(
        checkpoint.game_config,
        agent,
        opponent,
        games=args.games,
        swap_sides=not args.no_swap_sides,
    )
    payload = {
        "checkpoint": {
            "path": str(args.checkpoint),
            "iteration": checkpoint.iteration,
        },
        "opponent": opponent_metadata,
        "agent_simulations": args.simulations,
        "opponent_simulations": opponent_simulations,
        "mcts_simulations": args.mcts_simulations,
        "stats": stats.to_dict(),
    }
    if args.promotion_threshold is not None:
        payload["promotion"] = {
            "threshold": args.promotion_threshold,
            "passed": stats.agent_a_win_rate > args.promotion_threshold,
        }

    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

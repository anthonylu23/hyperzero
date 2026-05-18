"""Evaluate two baseline agents head to head.

Run from the repository root:

    python3 scripts/evaluate_baselines.py --agent-a heuristic --agent-b random
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyperzero.agents import HeuristicAgent, MCTSAgent, RandomAgent, TacticalAgent
from hyperzero.eval import evaluate_matchup
from hyperzero.game import GameConfig


def build_agent(name: str, seed: int, simulations: int):
    """Create a baseline agent by command-line name."""
    if name == "random":
        return RandomAgent(seed=seed, name=f"random-{seed}")
    if name == "tactical":
        return TacticalAgent(seed=seed, name=f"tactical-{seed}")
    if name == "heuristic":
        return HeuristicAgent(seed=seed, name=f"heuristic-{seed}")
    if name == "mcts":
        return MCTSAgent(
            simulations=simulations,
            seed=seed,
            name=f"mcts-{simulations}-{seed}",
        )
    raise ValueError(f"unknown agent: {name}")


def parse_shape(raw: str) -> tuple[int, ...]:
    """Parse comma-separated board dimensions."""
    try:
        shape = tuple(int(part) for part in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must contain integers") from exc
    if not shape or any(size <= 0 for size in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-a",
        choices=("random", "tactical", "heuristic", "mcts"),
        default="heuristic",
    )
    parser.add_argument(
        "--agent-b",
        choices=("random", "tactical", "heuristic", "mcts"),
        default="random",
    )
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--shape", type=parse_shape, default=(6, 7))
    parser.add_argument("--connect-k", type=int, default=4)
    parser.add_argument("--gravity-axis", type=int, default=0)
    parser.add_argument("--seed-a", type=int, default=1)
    parser.add_argument("--seed-b", type=int, default=2)
    parser.add_argument("--mcts-simulations", type=int, default=100)
    parser.add_argument("--no-swap-sides", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GameConfig(
        shape=args.shape,
        connect_k=args.connect_k,
        gravity_axis=args.gravity_axis,
    )
    agent_a = build_agent(args.agent_a, args.seed_a, args.mcts_simulations)
    agent_b = build_agent(args.agent_b, args.seed_b, args.mcts_simulations)
    stats = evaluate_matchup(
        config,
        agent_a,
        agent_b,
        games=args.games,
        swap_sides=not args.no_swap_sides,
    )
    print(json.dumps(stats.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

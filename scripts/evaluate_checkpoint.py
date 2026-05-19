#!/usr/bin/env python3
"""Evaluate a v1 neural checkpoint against a baseline or another checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hyperzero.agents import HeuristicAgent, MCTSAgent, RandomAgent, TacticalAgent
from hyperzero.eval import evaluate_matchup
from hyperzero.game.state import GameState
from hyperzero.search.puct import logits_to_policy
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
    parser.add_argument("--trace-losses-output", type=Path)
    parser.add_argument("--trace-max-games", type=int, default=8)
    parser.add_argument("--trace-top-actions", type=int, default=5)
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
    if args.trace_losses_output is not None:
        traces = trace_agent_a_losses(
            checkpoint.game_config,
            agent,
            stats.results,
            swap_sides=not args.no_swap_sides,
            max_games=args.trace_max_games,
            top_actions=args.trace_top_actions,
        )
        args.trace_losses_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_losses_output.write_text(
            json.dumps(traces, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload["loss_traces"] = {
            "path": str(args.trace_losses_output),
            "games": len(traces),
        }

    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n", encoding="utf-8")


def trace_agent_a_losses(
    config,
    agent,
    results,
    *,
    swap_sides: bool,
    max_games: int,
    top_actions: int,
) -> list[dict[str, Any]]:
    """Trace neural priors/value on games agent A lost."""
    if max_games <= 0:
        raise ValueError("trace_max_games must be positive")
    traces: list[dict[str, Any]] = []
    for game_index, result in enumerate(results):
        a_is_first = (not swap_sides) or game_index % 2 == 0
        agent_a_player = 1 if a_is_first else -1
        if result.winner in (0, agent_a_player):
            continue
        traces.append(
            trace_game_for_agent(
                config,
                agent,
                result.actions,
                game_index=game_index,
                agent_a_player=agent_a_player,
                winner=result.winner,
                top_actions=top_actions,
            )
        )
        if len(traces) >= max_games:
            break
    return traces


def trace_game_for_agent(
    config,
    agent,
    actions,
    *,
    game_index: int,
    agent_a_player: int,
    winner: int,
    top_actions: int,
) -> dict[str, Any]:
    state = GameState.new(config, use_line_counts=True)
    moves: list[dict[str, Any]] = []
    for ply, action in enumerate(actions):
        if state.player_to_move == agent_a_player:
            evaluation = agent.evaluator.evaluate(state)
            policy = logits_to_policy(evaluation.policy_logits, state.legal_mask())
            moves.append(
                {
                    "ply": ply,
                    "player": state.player_to_move,
                    "selected_action": int(action),
                    "selected_action_probability": float(policy[int(action)]),
                    "value_estimate": float(evaluation.value),
                    "policy_entropy": _policy_entropy(policy),
                    "top_actions": _top_policy_actions(policy, top_actions),
                    "immediate_winning_actions": _immediate_winning_actions(
                        state,
                        state.player_to_move,
                    ),
                    "opponent_immediate_wins_after_selected": (
                        _opponent_immediate_wins_after(state, int(action))
                    ),
                }
            )
        state.make_move(int(action))
    return {
        "game_index": game_index,
        "agent_a_player": agent_a_player,
        "winner": winner,
        "actions": [int(action) for action in actions],
        "agent_a_moves": moves,
    }


def _policy_entropy(policy: np.ndarray) -> float:
    positive = policy[policy > 0.0]
    return float(-(positive * np.log(positive)).sum())


def _top_policy_actions(policy: np.ndarray, top_actions: int) -> list[dict[str, Any]]:
    if top_actions <= 0:
        raise ValueError("trace_top_actions must be positive")
    ordered = np.argsort(policy)[::-1][:top_actions]
    return [
        {"action": int(action), "probability": float(policy[int(action)])}
        for action in ordered
        if policy[int(action)] > 0.0
    ]


def _immediate_winning_actions(state: GameState, player: int) -> list[int]:
    actions = []
    for action in state.legal_actions():
        action = int(action)
        state.make_move(action)
        try:
            if state.terminal and state.winner == player:
                actions.append(action)
        finally:
            state.undo_move()
    return actions


def _opponent_immediate_wins_after(state: GameState, action: int) -> list[int]:
    player = state.player_to_move
    state.make_move(action)
    try:
        return _immediate_winning_actions(state, -player)
    finally:
        state.undo_move()


if __name__ == "__main__":
    main()

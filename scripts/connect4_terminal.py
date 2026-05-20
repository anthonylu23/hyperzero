"""Terminal demo for 2D Connect Four.

Run from the repository root:

    /usr/local/bin/python3.12 scripts/connect4_terminal.py

Play against a baseline agent:

    python3 scripts/connect4_terminal.py --opponent heuristic
    python3 scripts/connect4_terminal.py --opponent mcts --human-player O

Play against a trained neural checkpoint:

    python3 scripts/connect4_terminal.py
        --opponent neural
        --checkpoint-dir /tmp/hyperzero-v1-full
    python3 scripts/connect4_terminal.py
        --opponent neural
        --checkpoint /tmp/hyperzero-v1-full/best_by_eval_score.pt

Watch two agents play:

    python3 scripts/connect4_terminal.py --x-agent heuristic --o-agent random
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyperzero.agents import (
    Agent,
    HeuristicAgent,
    MCTSAgent,
    RandomAgent,
    TacticalAgent,
)
from hyperzero.game import GameConfig, GameState, InvalidActionError
from hyperzero.training import LoadedCheckpoint, build_checkpoint_agent


PLAYER_MARKS = {
    0: ".",
    1: "X",
    -1: "O",
}

AGENT_CHOICES = ("none", "random", "tactical", "heuristic", "mcts", "neural")
DEFAULT_CONFIG = GameConfig(shape=(6, 7), connect_k=4, gravity_axis=0)


def render_board(state: GameState) -> None:
    """Print the board with the top row first."""
    board = state.board_tensor()
    rows, columns = state.config.shape

    print()
    for row in range(rows - 1, -1, -1):
        print(
            " ".join(
                PLAYER_MARKS[int(board[row, column])] for column in range(columns)
            )
        )
    print(" ".join(str(column) for column in range(columns)))
    print()


def read_action(state: GameState) -> int | None:
    """Read one column from stdin. Return None when the user quits."""
    player = PLAYER_MARKS[state.player_to_move]
    columns = state.config.action_shape[0]

    while True:
        raw = input(f"Player {player}, choose column 0-{columns - 1} or q to quit: ")
        raw = raw.strip().lower()
        if raw in {"q", "quit", "exit"}:
            return None

        try:
            action = int(raw)
        except ValueError:
            print("Enter a column number.")
            continue

        try:
            state.config.validate_action_index(action)
        except ValueError as exc:
            print(exc)
            continue

        if not state.legal_mask()[action]:
            print(f"Column {action} is full.")
            continue

        return action


def build_agent(
    name: str,
    *,
    seed: int,
    simulations: int,
    neural_simulations: int,
    c_puct: float,
    checkpoint_path: Path | None,
    device: str,
    player_mark: str,
) -> tuple[Agent | None, LoadedCheckpoint | None]:
    """Create the requested baseline agent."""
    if name == "none":
        return None, None
    if name == "random":
        return RandomAgent(seed=seed, name=f"{player_mark}-random"), None
    if name == "tactical":
        return TacticalAgent(seed=seed, name=f"{player_mark}-tactical"), None
    if name == "heuristic":
        return HeuristicAgent(seed=seed, name=f"{player_mark}-heuristic"), None
    if name == "mcts":
        return (
            MCTSAgent(
                simulations=simulations,
                seed=seed,
                name=f"{player_mark}-mcts",
            ),
            None,
        )
    if name == "neural":
        if checkpoint_path is None:
            raise ValueError("--checkpoint or --checkpoint-dir is required for neural")
        agent, checkpoint = build_checkpoint_agent(
            checkpoint_path,
            simulations=neural_simulations,
            c_puct=c_puct,
            device=device,
            seed=seed,
            name=f"{player_mark}-neural-{checkpoint_path.stem}",
        )
        return agent, checkpoint
    raise ValueError(f"unknown agent: {name}")


def resolve_checkpoint_path(
    checkpoint: Path | None,
    checkpoint_dir: Path | None,
) -> Path | None:
    """Resolve an explicit checkpoint or a directory's best checkpoint."""
    if checkpoint is not None and checkpoint_dir is not None:
        raise ValueError("use either --checkpoint or --checkpoint-dir, not both")
    if checkpoint is None and checkpoint_dir is None:
        return None

    path = checkpoint if checkpoint is not None else checkpoint_dir
    assert path is not None
    if path.is_dir():
        path = path / "best_by_eval_score.pt"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    if path.suffix != ".pt":
        raise ValueError(f"checkpoint must be a .pt file: {path}")
    return path


def player_from_mark(mark: str) -> int:
    """Return the player integer for an X/O mark."""
    return 1 if mark.upper() == "X" else -1


def select_action(
    state: GameState,
    agents: dict[int, Agent],
) -> int | None:
    """Select either a human-entered action or an agent action."""
    agent = agents.get(state.player_to_move)
    if agent is None:
        return read_action(state)

    action = agent.select_action(state)
    player = PLAYER_MARKS[state.player_to_move]
    print(f"Player {player} ({agent.name}) chooses {action}.")
    return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--opponent",
        choices=AGENT_CHOICES,
        default="none",
        help="Agent to play against. Default keeps two-human mode.",
    )
    parser.add_argument(
        "--human-player",
        choices=("X", "O"),
        default="X",
        help="Human side when an opponent agent is enabled.",
    )
    parser.add_argument(
        "--x-agent",
        choices=AGENT_CHOICES,
        default="none",
        help="Agent controlling X. Use none for a human player.",
    )
    parser.add_argument(
        "--o-agent",
        choices=AGENT_CHOICES,
        default="none",
        help="Agent controlling O. Use none for a human player.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--x-seed",
        type=int,
        default=None,
        help="Optional RNG seed for the X agent.",
    )
    parser.add_argument(
        "--o-seed",
        type=int,
        default=None,
        help="Optional RNG seed for the O agent.",
    )
    parser.add_argument("--mcts-simulations", type=int, default=100)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Neural checkpoint .pt file, or a directory containing "
            "best_by_eval_score.pt."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory containing best_by_eval_score.pt for neural play.",
    )
    parser.add_argument(
        "--neural-simulations",
        type=int,
        default=32,
        help="PUCT simulations per neural move.",
    )
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.opponent != "none" and (
        args.x_agent != "none" or args.o_agent != "none"
    ):
        parser.error("use either --opponent or --x-agent/--o-agent, not both")
    if args.neural_simulations <= 0:
        parser.error("--neural-simulations must be positive")
    try:
        args.resolved_checkpoint = resolve_checkpoint_path(
            args.checkpoint,
            args.checkpoint_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return args


def configure_agents(
    args: argparse.Namespace,
) -> tuple[dict[int, Agent], GameConfig]:
    """Build the configured X/O agent mapping."""
    x_agent_name = args.x_agent
    o_agent_name = args.o_agent

    if args.opponent != "none":
        human_player = player_from_mark(args.human_player)
        if human_player == 1:
            o_agent_name = args.opponent
        else:
            x_agent_name = args.opponent

    agents: dict[int, Agent] = {}
    x_agent, x_checkpoint = build_agent(
        x_agent_name,
        seed=args.seed if args.x_seed is None else args.x_seed,
        simulations=args.mcts_simulations,
        neural_simulations=args.neural_simulations,
        c_puct=args.c_puct,
        checkpoint_path=args.resolved_checkpoint,
        device=args.device,
        player_mark="X",
    )
    o_agent, o_checkpoint = build_agent(
        o_agent_name,
        seed=args.seed + 1 if args.o_seed is None else args.o_seed,
        simulations=args.mcts_simulations,
        neural_simulations=args.neural_simulations,
        c_puct=args.c_puct,
        checkpoint_path=args.resolved_checkpoint,
        device=args.device,
        player_mark="O",
    )
    if x_agent is not None:
        agents[1] = x_agent
    if o_agent is not None:
        agents[-1] = o_agent
    for agent in agents.values():
        agent.reset()

    checkpoints = [
        checkpoint for checkpoint in (x_checkpoint, o_checkpoint) if checkpoint
    ]
    config = DEFAULT_CONFIG
    if checkpoints:
        config = checkpoints[0].game_config
        if len(config.shape) != 2:
            raise ValueError(
                "the terminal demo can only render 2D checkpoints; "
                f"loaded checkpoint has shape={config.shape}"
            )
        if any(
            checkpoint.game_config.to_dict() != config.to_dict()
            for checkpoint in checkpoints
        ):
            raise ValueError("all neural agents must use the same game config")
    return agents, config


def describe_players(agents: dict[int, Agent]) -> None:
    """Print a compact summary of who controls each side."""
    x_controller = agents[1].name if 1 in agents else "human"
    o_controller = agents[-1].name if -1 in agents else "human"
    if not agents:
        print("Two-human mode. Player X goes first. Enter column numbers to play.")
        return

    print(f"X: {x_controller}. O: {o_controller}.")
    print("Player X goes first.")


def main() -> None:
    args = parse_args()
    try:
        agents, config = configure_agents(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    state = GameState.new(config, use_line_counts=True)

    print("HyperZero Connect Four terminal demo")
    print(
        "Board: "
        f"shape={config.shape}, connect_k={config.connect_k}, "
        f"gravity_axis={config.gravity_axis}"
    )
    describe_players(agents)

    while not state.terminal:
        render_board(state)
        action = select_action(state, agents)
        if action is None:
            print("Game ended by user.")
            return

        try:
            state.make_move(action)
        except InvalidActionError as exc:
            print(exc)

    render_board(state)
    if state.winner == 0:
        print("Draw.")
    else:
        print(f"Player {PLAYER_MARKS[state.winner]} wins.")

    print(f"Moves: {state.action_history()}")
    print(f"Hash: {int(state.zobrist_hash)}")


if __name__ == "__main__":
    main()

"""Terminal demo for 2D Connect Four.

Run from the repository root:

    /usr/local/bin/python3.12 scripts/connect4_terminal.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyperzero.game import GameConfig, GameState, InvalidActionError


PLAYER_MARKS = {
    0: ".",
    1: "X",
    -1: "O",
}


def render_board(state: GameState) -> None:
    """Print the board with the top row first."""
    board = state.board_tensor()
    rows, columns = state.config.shape

    print()
    for row in range(rows - 1, -1, -1):
        print(" ".join(PLAYER_MARKS[int(board[row, column])] for column in range(columns)))
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


def main() -> None:
    config = GameConfig(shape=(6, 7), connect_k=4, gravity_axis=0)
    state = GameState.new(config, use_line_counts=True)

    print("HyperZero Connect Four terminal demo")
    print("Player X goes first. Enter column numbers to play.")

    while not state.terminal:
        render_board(state)
        action = read_action(state)
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

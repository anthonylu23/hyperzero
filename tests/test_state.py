import numpy as np
import pytest

from hyperzero.game import GameConfig, GameState, InvalidActionError, TerminalStateError


def test_2d_vertical_win_with_gravity() -> None:
    config = GameConfig(shape=(6, 7), connect_k=4, gravity_axis=0)
    state = GameState.new(config)

    for action in [3, 4, 3, 4, 3, 4, 3]:
        state.make_move(action)

    assert state.terminal
    assert state.winner == 1
    assert state.last_move is not None
    assert state.last_move.cell_coord == (3, 3)
    assert state.terminal_value() == -1


def test_2d_horizontal_win() -> None:
    config = GameConfig(shape=(4, 4), connect_k=3, gravity_axis=0)
    state = GameState.new(config)

    for action in [0, 0, 1, 1, 2]:
        state.make_move(action)

    assert state.terminal
    assert state.winner == 1


def test_2d_diagonal_win() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)

    for action in [0, 1, 1, 2, 2, 0, 2]:
        state.make_move(action)

    assert state.terminal
    assert state.winner == 1
    assert state.last_move is not None
    assert state.last_move.cell_coord == (2, 2)


def test_3d_axis_win() -> None:
    config = GameConfig(shape=(3, 3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)
    action = config.action_index((1, 1))

    for move in [action, 0, action, 0, action]:
        state.make_move(move)

    assert state.terminal
    assert state.winner == 1
    assert state.last_move is not None
    assert state.last_move.cell_coord == (2, 1, 1)


def test_alternate_gravity_axis_places_along_that_axis() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=1)
    state = GameState.new(config)

    first = state.make_move(2)
    second = state.make_move(2)

    assert first.cell_coord == (2, 0)
    assert second.cell_coord == (2, 1)


def test_3d_space_diagonal_win_detection() -> None:
    config = GameConfig(shape=(3, 3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)
    target_action = config.action_index((2, 2))

    state.board[config.flat_index((0, 0, 0))] = 1
    state.board[config.flat_index((1, 1, 1))] = 1
    state.board[config.flat_index((0, 2, 2))] = -1
    state.board[config.flat_index((1, 2, 2))] = -1
    state.heights[target_action] = 2

    state.make_move(target_action)

    assert state.terminal
    assert state.winner == 1
    assert state.last_move is not None
    assert state.last_move.cell_coord == (2, 2, 2)


def test_full_column_invalid_move() -> None:
    config = GameConfig(shape=(2, 2), connect_k=2, gravity_axis=0)
    state = GameState.new(config)

    state.make_move(0)
    state.make_move(0)

    with pytest.raises(InvalidActionError):
        state.make_move(0)


def test_float_action_is_invalid() -> None:
    config = GameConfig(shape=(2, 2), connect_k=2, gravity_axis=0)
    state = GameState.new(config)

    with pytest.raises(InvalidActionError):
        state.make_move(1.5)  # type: ignore[arg-type]


def test_draw_detection() -> None:
    config = GameConfig(shape=(1, 2), connect_k=2, gravity_axis=0)
    state = GameState.new(config)

    state.make_move(0)
    state.make_move(1)

    assert state.terminal
    assert state.winner == 0
    assert state.terminal_value() == 0
    assert not state.legal_mask().any()


def test_canonical_board_flips_to_current_player() -> None:
    config = GameConfig(shape=(2, 2), connect_k=2, gravity_axis=0)
    state = GameState.new(config)

    move = state.make_move(1)
    canonical = state.canonical_board()

    assert state.player_to_move == -1
    assert state.board[move.cell_index] == 1
    assert canonical[move.cell_coord] == -1


def test_apply_returns_independent_next_state() -> None:
    config = GameConfig(shape=(2, 2), connect_k=2, gravity_axis=0)
    state = GameState.new(config)

    next_state = state.apply(1)

    assert state.ply == 0
    assert not state.board.any()
    assert next_state.ply == 1
    assert next_state.board.sum() == 1


def test_undo_restores_exact_state() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)
    state.make_move(0)

    board_before = state.board.copy()
    heights_before = state.heights.copy()
    player_before = state.player_to_move
    ply_before = state.ply
    last_move_before = state.last_move

    state.make_move(1)
    undone = state.undo_move()

    assert undone.action == 1
    np.testing.assert_array_equal(state.board, board_before)
    np.testing.assert_array_equal(state.heights, heights_before)
    assert state.player_to_move == player_before
    assert state.ply == ply_before
    assert state.last_move == last_move_before


def test_undo_after_terminal_win_restores_nonterminal_state() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)

    for action in [0, 1, 0, 1, 0]:
        state.make_move(action)

    assert state.terminal
    state.undo_move()

    assert not state.terminal
    assert state.winner is None
    assert state.legal_mask().any()


def test_hash_updates_and_restores_on_undo() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0, zobrist_seed=123)
    state = GameState.new(config)
    initial_hash = state.zobrist_hash

    state.make_move(0)
    state.make_move(1)
    mid_hash = state.zobrist_hash

    assert mid_hash != initial_hash
    assert state.recompute_hash() == mid_hash

    state.undo_move()
    state.undo_move()

    assert state.zobrist_hash == initial_hash
    assert state.recompute_hash() == initial_hash


def test_line_count_mode_detects_win_and_undoes_counts() -> None:
    config = GameConfig(shape=(4, 4), connect_k=3, gravity_axis=0)
    state = GameState.new(config, use_line_counts=True)

    for action in [0, 0, 1, 1, 2]:
        state.make_move(action)

    assert state.terminal
    assert state.winner == 1
    assert state.line_counts is not None
    assert state.line_counts.max() == 3

    state.undo_move()

    assert not state.terminal
    assert state.line_counts.max() == 2


def test_snapshot_contains_gui_safe_fields() -> None:
    config = GameConfig(shape=(2, 2), connect_k=2, gravity_axis=0)
    state = GameState.new(config)
    state.make_move(1)

    snapshot = state.to_snapshot()

    assert snapshot["shape"] == (2, 2)
    assert snapshot["player_to_move"] == -1
    assert snapshot["hash"] == int(state.zobrist_hash)
    assert snapshot["last_move"] == {
        "action": 1,
        "column_coord": (1,),
        "cell_index": 1,
        "cell_coord": (0, 1),
        "player": 1,
        "ply": 0,
    }


def test_terminal_state_rejects_moves() -> None:
    config = GameConfig(shape=(1, 2), connect_k=2, gravity_axis=0)
    state = GameState.new(config)
    state.make_move(0)
    state.make_move(1)

    with pytest.raises(TerminalStateError):
        state.make_move(0)

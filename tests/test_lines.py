import numpy as np

from hyperzero.game.lines import generate_directions, generate_winning_lines


def test_generate_directions_removes_opposite_duplicates() -> None:
    assert set(generate_directions(2)) == {
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    }
    assert len(generate_directions(3)) == 13


def test_generate_winning_lines_includes_2d_diagonals() -> None:
    lines = generate_winning_lines((3, 3), 3)
    coords = {
        tuple(np.unravel_index(cell, (3, 3)) for cell in line)
        for line in lines
    }

    assert ((0, 0), (1, 1), (2, 2)) in coords
    assert ((0, 2), (1, 1), (2, 0)) in coords


def test_generate_winning_lines_includes_3d_space_diagonal() -> None:
    lines = generate_winning_lines((3, 3, 3), 3)
    coords = {
        tuple(np.unravel_index(cell, (3, 3, 3)) for cell in line)
        for line in lines
    }

    assert ((0, 0, 0), (1, 1, 1), (2, 2, 2)) in coords
    assert ((0, 0, 2), (1, 1, 1), (2, 2, 0)) in coords


def test_generate_winning_lines_includes_4d_hyperdiagonal() -> None:
    lines = generate_winning_lines((3, 3, 3, 3), 3)
    coords = {
        tuple(np.unravel_index(cell, (3, 3, 3, 3)) for cell in line)
        for line in lines
    }

    assert ((0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)) in coords


def test_generate_winning_lines_deduplicates_connect_one_cells() -> None:
    lines = generate_winning_lines((2, 2), 1)

    assert lines.shape == (4, 1)
    assert {tuple(line) for line in lines} == {(0,), (1,), (2,), (3,)}

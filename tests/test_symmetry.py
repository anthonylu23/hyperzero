import numpy as np

from hyperzero.game import GameConfig
from hyperzero.game.symmetry import gravity_preserving_symmetries


def test_2d_symmetries_include_horizontal_reflection_only() -> None:
    config = GameConfig(shape=(6, 7), connect_k=4, gravity_axis=0)
    symmetries = gravity_preserving_symmetries(config)

    assert len(symmetries) == 2
    reflected = next(symmetry for symmetry in symmetries if symmetry.flips == (True,))

    assert reflected.transform_action(0) == 6
    assert reflected.transform_action(6) == 0


def test_3d_cube_has_eight_gravity_preserving_symmetries() -> None:
    config = GameConfig(shape=(4, 4, 4), connect_k=4, gravity_axis=0)
    symmetries = gravity_preserving_symmetries(config)

    assert len(symmetries) == 8


def test_non_cubic_3d_symmetries_do_not_swap_unequal_axes() -> None:
    config = GameConfig(shape=(4, 2, 3), connect_k=4, gravity_axis=0)
    symmetries = gravity_preserving_symmetries(config)

    assert len(symmetries) == 4
    assert {symmetry.permutation for symmetry in symmetries} == {(0, 1)}


def test_symmetry_transforms_board_and_policy_consistently() -> None:
    config = GameConfig(shape=(2, 3), connect_k=2, gravity_axis=0)
    reflected = next(
        symmetry
        for symmetry in gravity_preserving_symmetries(config)
        if symmetry.flips == (True,)
    )
    board = np.zeros(config.shape, dtype=np.int8)
    board[0, 0] = 1
    policy = np.array([0.1, 0.2, 0.7])

    transformed_board = reflected.transform_board(board)
    transformed_policy = reflected.transform_policy(policy)

    assert transformed_board[0, 2] == 1
    np.testing.assert_allclose(transformed_policy, np.array([0.7, 0.2, 0.1]))

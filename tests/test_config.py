import numpy as np
import pytest

from hyperzero.game import GameConfig


def test_config_builds_3d_action_and_column_mappings() -> None:
    config = GameConfig(shape=(4, 4, 4), connect_k=4, gravity_axis=0)

    assert config.num_cells == 64
    assert config.action_shape == (4, 4)
    assert config.num_actions == 16
    assert config.action_index((2, 3)) == 11
    assert config.column_coord(11) == (2, 3)

    cells = config.column_cells[11]
    assert [config.cell_coord(cell) for cell in cells] == [
        (0, 2, 3),
        (1, 2, 3),
        (2, 2, 3),
        (3, 2, 3),
    ]


def test_lines_by_cell_references_only_containing_lines() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    center = config.flat_index((1, 1))

    assert config.lines_by_cell[center]
    for line_id in config.lines_by_cell[center]:
        assert center in config.winning_lines[line_id]

    assert config.winning_lines.dtype == np.int32


def test_invalid_configs_are_rejected() -> None:
    with pytest.raises(ValueError):
        GameConfig(shape=(), connect_k=1)
    with pytest.raises(ValueError):
        GameConfig(shape=(0, 3), connect_k=1)
    with pytest.raises(ValueError):
        GameConfig(shape=(3, 3), connect_k=0)
    with pytest.raises(ValueError):
        GameConfig(shape=(3, 3), connect_k=4)
    with pytest.raises(ValueError):
        GameConfig(shape=(3, 3), connect_k=3, gravity_axis=2)


def test_config_round_trips_through_dict() -> None:
    config = GameConfig(shape=(4, 5, 6), connect_k=4, gravity_axis=2, zobrist_seed=17)
    restored = GameConfig.from_dict(config.to_dict())

    assert restored.shape == config.shape
    assert restored.connect_k == config.connect_k
    assert restored.gravity_axis == config.gravity_axis
    assert restored.zobrist_seed == config.zobrist_seed
    np.testing.assert_array_equal(restored.zobrist_pieces, config.zobrist_pieces)

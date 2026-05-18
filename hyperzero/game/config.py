"""Immutable game configuration and precomputed lookup tables."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hyperzero.game.lines import build_lines_by_cell, generate_winning_lines


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Rules and lookup tables for one N-dimensional Connect-K variant."""

    shape: tuple[int, ...]
    connect_k: int
    gravity_axis: int = 0
    zobrist_seed: int = 0
    num_cells: int = field(init=False)
    gravity_size: int = field(init=False)
    action_axes: tuple[int, ...] = field(init=False)
    action_shape: tuple[int, ...] = field(init=False)
    num_actions: int = field(init=False)
    action_coords: tuple[tuple[int, ...], ...] = field(init=False)
    column_cells: np.ndarray = field(init=False, repr=False)
    winning_lines: np.ndarray = field(init=False, repr=False)
    lines_by_cell: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    zobrist_pieces: np.ndarray = field(init=False, repr=False)
    zobrist_side: np.uint64 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        shape = tuple(int(size) for size in self.shape)
        connect_k = int(self.connect_k)
        gravity_axis = int(self.gravity_axis)
        zobrist_seed = int(self.zobrist_seed)

        if not shape:
            raise ValueError("shape must contain at least one dimension")
        if any(size <= 0 for size in shape):
            raise ValueError("shape dimensions must be positive")
        if connect_k <= 0:
            raise ValueError("connect_k must be positive")
        if connect_k > max(shape):
            raise ValueError("connect_k must fit within at least one board dimension")
        if not 0 <= gravity_axis < len(shape):
            raise ValueError("gravity_axis is out of range")

        action_axes = tuple(axis for axis in range(len(shape)) if axis != gravity_axis)
        action_shape = tuple(shape[axis] for axis in action_axes)
        action_coords = tuple(
            tuple(int(value) for value in coord)
            for coord in np.ndindex(action_shape)
        )

        if not action_coords:
            action_coords = ((),)

        num_cells = int(np.prod(shape, dtype=np.int64))
        gravity_size = shape[gravity_axis]
        num_actions = len(action_coords)
        column_cells = np.empty((num_actions, gravity_size), dtype=np.int32)

        for action, action_coord in enumerate(action_coords):
            full_coord = [0] * len(shape)
            for axis, value in zip(action_axes, action_coord, strict=True):
                full_coord[axis] = value

            for gravity_coord in range(gravity_size):
                full_coord[gravity_axis] = gravity_coord
                column_cells[action, gravity_coord] = np.ravel_multi_index(
                    tuple(full_coord),
                    shape,
                )

        winning_lines = generate_winning_lines(shape, connect_k)
        lines_by_cell = build_lines_by_cell(num_cells, winning_lines)
        rng = np.random.default_rng(zobrist_seed)
        zobrist_pieces = rng.integers(
            1,
            np.iinfo(np.uint64).max,
            size=(num_cells, 2),
            dtype=np.uint64,
        )
        zobrist_side = np.uint64(
            rng.integers(1, np.iinfo(np.uint64).max, dtype=np.uint64)
        )

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "connect_k", connect_k)
        object.__setattr__(self, "gravity_axis", gravity_axis)
        object.__setattr__(self, "zobrist_seed", zobrist_seed)
        object.__setattr__(self, "num_cells", num_cells)
        object.__setattr__(self, "gravity_size", gravity_size)
        object.__setattr__(self, "action_axes", action_axes)
        object.__setattr__(self, "action_shape", action_shape)
        object.__setattr__(self, "num_actions", num_actions)
        object.__setattr__(self, "action_coords", action_coords)
        object.__setattr__(self, "column_cells", column_cells)
        object.__setattr__(self, "winning_lines", winning_lines)
        object.__setattr__(self, "lines_by_cell", lines_by_cell)
        object.__setattr__(self, "zobrist_pieces", zobrist_pieces)
        object.__setattr__(self, "zobrist_side", zobrist_side)

    def action_index(self, column_coord: tuple[int, ...]) -> int:
        """Return the flat action id for a non-gravity column coordinate."""
        if self.action_shape == ():
            if column_coord != ():
                raise ValueError("1D games use an empty column coordinate")
            return 0
        return int(np.ravel_multi_index(column_coord, self.action_shape))

    def column_coord(self, action: int) -> tuple[int, ...]:
        """Return the non-gravity column coordinate for a flat action id."""
        self.validate_action_index(action)
        return self.action_coords[int(action)]

    def flat_index(self, coord: tuple[int, ...]) -> int:
        """Return the flat cell index for a full board coordinate."""
        return int(np.ravel_multi_index(coord, self.shape))

    def cell_coord(self, flat_index: int) -> tuple[int, ...]:
        """Return the full board coordinate for a flat cell index."""
        return tuple(
            int(value) for value in np.unravel_index(int(flat_index), self.shape)
        )

    def validate_action_index(self, action: int) -> None:
        """Raise ValueError if action is outside the action space."""
        if not 0 <= int(action) < self.num_actions:
            raise ValueError(f"action {action} is outside [0, {self.num_actions})")

    def zobrist_piece(self, cell_index: int, player: int) -> np.uint64:
        """Return the Zobrist key for a player occupying one flat cell."""
        if player not in (-1, 1):
            raise ValueError("player must be 1 or -1")
        player_index = 1 if player == 1 else 0
        return np.uint64(self.zobrist_pieces[int(cell_index), player_index])

    def to_dict(self) -> dict[str, object]:
        """Return the serializable configuration fields."""
        return {
            "shape": self.shape,
            "connect_k": self.connect_k,
            "gravity_axis": self.gravity_axis,
            "zobrist_seed": self.zobrist_seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GameConfig:
        """Build a config from a serialized config dictionary."""
        return cls(
            shape=tuple(int(size) for size in data["shape"]),  # type: ignore[index]
            connect_k=int(data["connect_k"]),
            gravity_axis=int(data.get("gravity_axis", 0)),
            zobrist_seed=int(data.get("zobrist_seed", 0)),
        )

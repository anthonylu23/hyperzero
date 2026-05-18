"""Gravity-preserving board and policy symmetries."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np

from hyperzero.game.config import GameConfig


@dataclass(frozen=True, slots=True)
class Symmetry:
    """A transformation over non-gravity action axes."""

    config: GameConfig
    permutation: tuple[int, ...]
    flips: tuple[bool, ...]

    def transform_action_coord(self, action_coord: tuple[int, ...]) -> tuple[int, ...]:
        """Transform a non-gravity column coordinate."""
        if len(action_coord) != len(self.config.action_shape):
            raise ValueError("action coordinate rank does not match config")

        transformed = []
        for output_axis, source_axis in enumerate(self.permutation):
            value = int(action_coord[source_axis])
            size = self.config.action_shape[output_axis]
            if self.flips[output_axis]:
                value = size - 1 - value
            transformed.append(value)
        return tuple(transformed)

    def transform_action(self, action: int) -> int:
        """Transform a flat action id."""
        return self.config.action_index(
            self.transform_action_coord(self.config.column_coord(action))
        )

    def transform_policy(self, policy: np.ndarray) -> np.ndarray:
        """Transform a flat policy or action-shaped tensor."""
        policy = np.asarray(policy)
        original_shape = policy.shape
        flat_policy = policy.reshape(self.config.num_actions)
        transformed = np.empty_like(flat_policy)
        for action, probability in enumerate(flat_policy):
            transformed[self.transform_action(action)] = probability
        if original_shape == self.config.action_shape:
            return transformed.reshape(self.config.action_shape)
        return transformed

    def transform_board(self, board: np.ndarray) -> np.ndarray:
        """Transform a board in either flat or config-shaped form."""
        board = np.asarray(board)
        original_shape = board.shape
        flat_board = board.reshape(self.config.num_cells)
        transformed = np.empty_like(flat_board)

        for flat_cell, value in enumerate(flat_board):
            coord = self.config.cell_coord(flat_cell)
            action_coord = tuple(coord[axis] for axis in self.config.action_axes)
            transformed_action = self.transform_action_coord(action_coord)
            transformed_coord = list(coord)
            for axis, transformed_value in zip(
                self.config.action_axes,
                transformed_action,
                strict=True,
            ):
                transformed_coord[axis] = transformed_value
            transformed[self.config.flat_index(tuple(transformed_coord))] = value

        if original_shape == self.config.shape:
            return transformed.reshape(self.config.shape)
        return transformed


def gravity_preserving_symmetries(config: GameConfig) -> tuple[Symmetry, ...]:
    """Return symmetries that keep the gravity axis fixed."""
    rank = len(config.action_shape)
    if rank == 0:
        return (Symmetry(config=config, permutation=(), flips=()),)

    valid_permutations = []
    for permutation in permutations(range(rank)):
        if all(
            config.action_shape[output_axis] == config.action_shape[source_axis]
            for output_axis, source_axis in enumerate(permutation)
        ):
            valid_permutations.append(tuple(int(axis) for axis in permutation))

    symmetries = []
    for permutation in valid_permutations:
        for flips in product((False, True), repeat=rank):
            symmetries.append(
                Symmetry(
                    config=config,
                    permutation=permutation,
                    flips=tuple(bool(flip) for flip in flips),
                )
            )
    return tuple(symmetries)

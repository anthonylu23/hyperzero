"""Winning-line generation for N-dimensional Connect-K."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import product

import numpy as np


def generate_directions(dimensions: int) -> tuple[tuple[int, ...], ...]:
    """Return unique N-dimensional directions with components in {-1, 0, 1}."""
    if dimensions < 1:
        raise ValueError("dimensions must be at least 1")

    directions: list[tuple[int, ...]] = []
    for direction in product((-1, 0, 1), repeat=dimensions):
        if all(component == 0 for component in direction):
            continue

        first_nonzero = next(component for component in direction if component != 0)
        if first_nonzero > 0:
            directions.append(tuple(int(component) for component in direction))

    return tuple(directions)


def generate_winning_lines(shape: Iterable[int], connect_k: int) -> np.ndarray:
    """Generate every length-K winning segment as flat cell indices."""
    shape_tuple = tuple(int(size) for size in shape)
    if not shape_tuple:
        raise ValueError("shape must contain at least one dimension")
    if any(size <= 0 for size in shape_tuple):
        raise ValueError("shape dimensions must be positive")
    if connect_k <= 0:
        raise ValueError("connect_k must be positive")

    dimensions = len(shape_tuple)
    lines: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    for direction in generate_directions(dimensions):
        ranges = []
        for axis, step in enumerate(direction):
            size = shape_tuple[axis]
            if step > 0:
                ranges.append(range(0, size - connect_k + 1))
            elif step < 0:
                ranges.append(range(connect_k - 1, size))
            else:
                ranges.append(range(0, size))

        if any(len(axis_range) == 0 for axis_range in ranges):
            continue

        for start in product(*ranges):
            cells = []
            for offset in range(connect_k):
                coord = tuple(
                    start[axis] + offset * direction[axis]
                    for axis in range(dimensions)
                )
                cells.append(int(np.ravel_multi_index(coord, shape_tuple)))
            line = tuple(cells)
            if line not in seen:
                seen.add(line)
                lines.append(line)

    if not lines:
        return np.empty((0, connect_k), dtype=np.int32)

    return np.asarray(lines, dtype=np.int32)


def build_lines_by_cell(
    num_cells: int,
    winning_lines: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    """Return line ids containing each flat cell index."""
    if num_cells <= 0:
        raise ValueError("num_cells must be positive")

    line_ids: list[list[int]] = [[] for _ in range(num_cells)]
    for line_id, line in enumerate(winning_lines):
        for cell in line:
            line_ids[int(cell)].append(line_id)

    return tuple(tuple(ids) for ids in line_ids)

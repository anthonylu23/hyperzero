"""Simple engine benchmarks for move generation and random playouts.

Run from the repository root:

    python3 benchmarks/benchmark_engine.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyperzero.game import GameConfig, GameState


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    shape: tuple[int, ...]
    connect_k: int
    gravity_axis: int = 0


VARIANTS = (
    Variant("2D 6x7 K4", (6, 7), 4),
    Variant("3D 4x4x4 K4", (4, 4, 4), 4),
    Variant("4D 4x4x4x4 K4", (4, 4, 4, 4), 4),
)


def benchmark_move_undo(
    config: GameConfig,
    *,
    iterations: int,
    use_line_counts: bool,
) -> float:
    """Return make_move/undo operations per second."""
    state = GameState.new(config, use_line_counts=use_line_counts)
    action = 0
    start = perf_counter()
    for _ in range(iterations):
        state.make_move(action)
        state.undo_move()
    elapsed = perf_counter() - start
    return iterations / elapsed


def benchmark_legal_mask(config: GameConfig, *, iterations: int) -> float:
    """Return legal-mask generations per second."""
    state = GameState.new(config)
    start = perf_counter()
    for _ in range(iterations):
        state.legal_mask()
    elapsed = perf_counter() - start
    return iterations / elapsed


def benchmark_random_playouts(
    config: GameConfig,
    *,
    games: int,
    use_line_counts: bool,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return games per second and average game length."""
    plies = 0
    start = perf_counter()
    for _ in range(games):
        state = GameState.new(config, use_line_counts=use_line_counts)
        while not state.terminal:
            legal_actions = state.legal_actions()
            action = int(rng.choice(legal_actions))
            state.make_move(action)
        plies += state.ply
    elapsed = perf_counter() - start
    return games / elapsed, plies / games


def main() -> None:
    rng = np.random.default_rng(0)
    print(
        "variant,lines,legal_masks/s,move_undo/s,move_undo_line_counts/s,"
        "random_games/s,random_games_line_counts/s,avg_len,avg_len_line_counts"
    )
    for variant in VARIANTS:
        config = GameConfig(
            shape=variant.shape,
            connect_k=variant.connect_k,
            gravity_axis=variant.gravity_axis,
        )
        legal_masks = benchmark_legal_mask(config, iterations=50_000)
        move_undo = benchmark_move_undo(
            config,
            iterations=20_000,
            use_line_counts=False,
        )
        move_undo_counts = benchmark_move_undo(
            config,
            iterations=20_000,
            use_line_counts=True,
        )
        games, avg_len = benchmark_random_playouts(
            config,
            games=500,
            use_line_counts=False,
            rng=rng,
        )
        games_counts, avg_len_counts = benchmark_random_playouts(
            config,
            games=500,
            use_line_counts=True,
            rng=rng,
        )
        print(
            f"{variant.name},{len(config.winning_lines)},"
            f"{legal_masks:.0f},{move_undo:.0f},{move_undo_counts:.0f},"
            f"{games:.1f},{games_counts:.1f},{avg_len:.1f},{avg_len_counts:.1f}"
        )


if __name__ == "__main__":
    main()

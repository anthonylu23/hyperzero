"""Self-play helpers for universal mixed-variant training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hyperzero.game.config import GameConfig
from hyperzero.search.puct import PolicyValueEvaluator, PUCTConfig
from hyperzero.training.self_play import (
    _ActiveSelfPlayGame,
    _complete_leaves_batched,
    _initialize_search_sessions_batched,
    generate_game,
    generate_games_batched,
)
from hyperzero.training.universal_replay import UniversalSelfPlayExample


@dataclass(frozen=True, slots=True)
class UniversalGameSpec:
    """Named game variant used by a universal training run."""

    config_id: str
    game_config: GameConfig
    self_play_games_per_iteration: int

    def __post_init__(self) -> None:
        if not self.config_id:
            raise ValueError("config_id must be nonempty")
        if self.self_play_games_per_iteration <= 0:
            raise ValueError("self_play_games_per_iteration must be positive")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/checkpoint serializable representation."""
        return {
            "config_id": self.config_id,
            "game_config": self.game_config.to_dict(),
            "self_play_games_per_iteration": self.self_play_games_per_iteration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> UniversalGameSpec:
        """Build a game spec from serialized data."""
        return cls(
            config_id=str(data["config_id"]),
            game_config=GameConfig.from_dict(data["game_config"]),  # type: ignore[arg-type]
            self_play_games_per_iteration=int(data["self_play_games_per_iteration"]),
        )


def generate_universal_examples(
    spec: UniversalGameSpec,
    evaluator: PolicyValueEvaluator,
    *,
    search_config: PUCTConfig,
    rng: np.random.Generator,
    batched_self_play: bool = False,
    max_active_games: int | None = None,
) -> tuple[list[UniversalSelfPlayExample], int, float]:
    """Generate examples for one game spec and return examples/games/avg length."""
    if batched_self_play:
        games = generate_games_batched(
            spec.game_config,
            evaluator,
            games=spec.self_play_games_per_iteration,
            search_config=search_config,
            rng=rng,
            max_active_games=max_active_games,
            use_line_counts=True,
        )
    else:
        games = tuple(
            generate_game(
                spec.game_config,
                evaluator,
                search_config=search_config,
                rng=rng,
                use_line_counts=True,
            )
            for _ in range(spec.self_play_games_per_iteration)
        )

    examples: list[UniversalSelfPlayExample] = []
    for game in games:
        for example in game.examples:
            examples.append(
                UniversalSelfPlayExample(
                    config_id=spec.config_id,
                    game_config=spec.game_config,
                    board=example.board,
                    policy=example.policy,
                    value=example.value,
                    legal_mask=example.legal_mask,
                    player_to_move=example.player_to_move,
                    ply=example.ply,
                )
            )
    average_length = (
        float(np.mean([len(game.actions) for game in games])) if games else 0.0
    )
    return examples, len(games), average_length


def generate_universal_examples_batched_across_specs(
    specs: tuple[UniversalGameSpec, ...],
    evaluator: PolicyValueEvaluator,
    *,
    search_config: PUCTConfig,
    rng: np.random.Generator,
    max_active_games: int | None = None,
) -> tuple[list[UniversalSelfPlayExample], dict[str, int], dict[str, float]]:
    """Generate mixed-variant self-play while batching inference across variants."""
    if not specs:
        raise ValueError("specs must be nonempty")
    total_games = sum(spec.self_play_games_per_iteration for spec in specs)
    if total_games <= 0:
        raise ValueError("total games must be positive")
    max_active_games = total_games if max_active_games is None else max_active_games
    if max_active_games <= 0:
        raise ValueError("max_active_games must be positive")

    completed: dict[str, list[_ActiveSelfPlayGame]] = {
        spec.config_id: [] for spec in specs
    }
    active: list[tuple[UniversalGameSpec, _ActiveSelfPlayGame]] = []
    started_by_config = {spec.config_id: 0 for spec in specs}
    spec_cursor = 0

    def start_games() -> None:
        nonlocal spec_cursor
        attempts = 0
        while (
            len(active) < max_active_games
            and sum(started_by_config.values()) < total_games
        ):
            spec = specs[spec_cursor % len(specs)]
            spec_cursor += 1
            attempts += 1
            if started_by_config[spec.config_id] >= spec.self_play_games_per_iteration:
                if attempts > len(specs) * 2:
                    break
                continue
            active.append(
                (
                    spec,
                    _ActiveSelfPlayGame(
                        spec.game_config,
                        np.random.default_rng(int(rng.integers(2**63 - 1))),
                        True,
                    ),
                )
            )
            started_by_config[spec.config_id] += 1
            attempts = 0

    start_games()
    while active:
        sessions = [
            (spec, game, game.build_search_session(search_config))
            for spec, game in active
            if not game.terminal
        ]
        _initialize_search_sessions_batched(
            [session for _, _, session in sessions],
            evaluator,
        )

        while sessions:
            leaves_by_session = [
                (session, session.select_leaf())
                for _, _, session in sessions
                if not session.complete
            ]
            if not leaves_by_session:
                break
            _complete_leaves_batched(leaves_by_session, evaluator)

        still_active: list[tuple[UniversalGameSpec, _ActiveSelfPlayGame]] = []
        for spec, game, session in sessions:
            game.apply_search_result(session.result().policy)
            if game.terminal:
                completed[spec.config_id].append(game)
            else:
                still_active.append((spec, game))
        active = still_active
        start_games()

    examples: list[UniversalSelfPlayExample] = []
    game_counts: dict[str, int] = {}
    average_lengths: dict[str, float] = {}
    specs_by_id = {spec.config_id: spec for spec in specs}
    for config_id, games in completed.items():
        spec = specs_by_id[config_id]
        game_counts[config_id] = len(games)
        average_lengths[config_id] = (
            float(np.mean([len(game.actions) for game in games])) if games else 0.0
        )
        for game in games:
            finalized = game.finalize()
            for example in finalized.examples:
                examples.append(
                    UniversalSelfPlayExample(
                        config_id=config_id,
                        game_config=spec.game_config,
                        board=example.board,
                        policy=example.policy,
                        value=example.value,
                        legal_mask=example.legal_mask,
                        player_to_move=example.player_to_move,
                        ply=example.ply,
                    )
                )
    return examples, game_counts, average_lengths

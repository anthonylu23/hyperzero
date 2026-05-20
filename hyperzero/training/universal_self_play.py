"""Self-play helpers for universal mixed-variant training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hyperzero.game.config import GameConfig
from hyperzero.search.puct import PolicyValueEvaluator, PUCTConfig
from hyperzero.training.self_play import generate_game, generate_games_batched
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
        )
    else:
        games = tuple(
            generate_game(
                spec.game_config,
                evaluator,
                search_config=search_config,
                rng=rng,
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

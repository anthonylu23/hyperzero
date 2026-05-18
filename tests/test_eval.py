from dataclasses import dataclass, field

from hyperzero.agents import RandomAgent
from hyperzero.eval import evaluate_matchup, play_game
from hyperzero.game import GameConfig, GameState


@dataclass(slots=True)
class FixedAgent:
    actions: list[int]
    name: str
    index: int = field(default=0, init=False)

    def reset(self) -> None:
        self.index = 0

    def select_action(self, state: GameState) -> int:
        action = self.actions[self.index]
        self.index += 1
        return action


def test_play_game_returns_replay_and_move_metrics() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    first = FixedAgent([0, 0, 0], "first")
    second = FixedAgent([1, 1], "second")

    result = play_game(config, first, second)

    assert result.winner == 1
    assert result.ply == 5
    assert result.actions == (0, 1, 0, 1, 0)
    assert len(result.move_times) == 5
    assert result.replay.playback().winner == 1


def test_evaluate_matchup_reports_totals() -> None:
    config = GameConfig(shape=(2, 2), connect_k=2, gravity_axis=0)
    stats = evaluate_matchup(
        config,
        RandomAgent(seed=1, name="a"),
        RandomAgent(seed=2, name="b"),
        games=4,
    )

    assert stats.games == 4
    assert stats.agent_a_wins + stats.agent_b_wins + stats.draws == 4
    assert stats.average_game_length > 0
    assert stats.average_move_time >= 0.0
    assert stats.runtime_seconds >= 0.0
    assert stats.to_dict()["games"] == 4

"""Head-to-head evaluation for Connect-K agents."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from hyperzero.agents.base import Agent
from hyperzero.game.config import GameConfig
from hyperzero.game.errors import InvalidActionError
from hyperzero.game.replay import GameReplay
from hyperzero.game.state import GameState


@dataclass(frozen=True, slots=True)
class GameResult:
    """Result of one completed head-to-head game."""

    winner: int
    ply: int
    actions: tuple[int, ...]
    replay: GameReplay
    move_times: tuple[float, ...]
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class MatchupStats:
    """Aggregate results from agent_a's perspective."""

    games: int
    agent_a_wins: int
    agent_b_wins: int
    draws: int
    average_game_length: float
    average_move_time: float
    runtime_seconds: float
    results: tuple[GameResult, ...]

    @property
    def agent_a_win_rate(self) -> float:
        """Return non-draw win rate for agent A over all games."""
        return self.agent_a_wins / self.games if self.games else 0.0

    @property
    def agent_b_win_rate(self) -> float:
        """Return non-draw win rate for agent B over all games."""
        return self.agent_b_wins / self.games if self.games else 0.0

    @property
    def draw_rate(self) -> float:
        """Return draw rate over all games."""
        return self.draws / self.games if self.games else 0.0

    def to_dict(self) -> dict[str, float | int]:
        """Return JSON-friendly aggregate metrics."""
        return {
            "games": self.games,
            "agent_a_wins": self.agent_a_wins,
            "agent_b_wins": self.agent_b_wins,
            "draws": self.draws,
            "agent_a_win_rate": self.agent_a_win_rate,
            "agent_b_win_rate": self.agent_b_win_rate,
            "draw_rate": self.draw_rate,
            "average_game_length": self.average_game_length,
            "average_move_time": self.average_move_time,
            "runtime_seconds": self.runtime_seconds,
        }


def play_game(
    config: GameConfig,
    first_agent: Agent,
    second_agent: Agent,
    *,
    use_line_counts: bool = True,
) -> GameResult:
    """Play one game with first_agent as player 1 and second_agent as player -1."""
    state = GameState.new(config, use_line_counts=use_line_counts)
    agents = {1: first_agent, -1: second_agent}
    for agent in agents.values():
        agent.reset()

    move_times: list[float] = []
    start = perf_counter()
    while not state.terminal:
        agent = agents[state.player_to_move]
        move_start = perf_counter()
        action = int(agent.select_action(state))
        move_times.append(perf_counter() - move_start)
        legal_mask = state.legal_mask()
        if action < 0 or action >= state.config.num_actions or not legal_mask[action]:
            raise InvalidActionError(
                f"{agent.name} selected illegal action {action} at ply {state.ply}"
            )
        state.make_move(action)

    runtime = perf_counter() - start
    replay = GameReplay.from_state(
        state,
        metadata={
            "player_1_agent": first_agent.name,
            "player_-1_agent": second_agent.name,
        },
    )
    return GameResult(
        winner=0 if state.winner is None else state.winner,
        ply=state.ply,
        actions=state.action_history(),
        replay=replay,
        move_times=tuple(move_times),
        runtime_seconds=runtime,
    )


def evaluate_matchup(
    config: GameConfig,
    agent_a: Agent,
    agent_b: Agent,
    *,
    games: int,
    swap_sides: bool = True,
    use_line_counts: bool = True,
) -> MatchupStats:
    """Evaluate two agents over repeated games and return aggregate metrics."""
    if games <= 0:
        raise ValueError("games must be positive")

    results: list[GameResult] = []
    agent_a_wins = 0
    agent_b_wins = 0
    draws = 0
    start = perf_counter()

    for game_index in range(games):
        a_is_first = (not swap_sides) or game_index % 2 == 0
        first_agent = agent_a if a_is_first else agent_b
        second_agent = agent_b if a_is_first else agent_a
        result = play_game(
            config,
            first_agent,
            second_agent,
            use_line_counts=use_line_counts,
        )
        results.append(result)

        if result.winner == 0:
            draws += 1
            continue

        winner_is_a = (result.winner == 1 and a_is_first) or (
            result.winner == -1 and not a_is_first
        )
        if winner_is_a:
            agent_a_wins += 1
        else:
            agent_b_wins += 1

    runtime = perf_counter() - start
    total_moves = sum(len(result.move_times) for result in results)
    total_move_time = sum(float(np.sum(result.move_times)) for result in results)
    return MatchupStats(
        games=games,
        agent_a_wins=agent_a_wins,
        agent_b_wins=agent_b_wins,
        draws=draws,
        average_game_length=float(np.mean([result.ply for result in results])),
        average_move_time=total_move_time / total_moves if total_moves else 0.0,
        runtime_seconds=runtime,
        results=tuple(results),
    )

"""V1 self-play data generation for neural-guided search."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hyperzero.game.actions import sample_action
from hyperzero.game.config import GameConfig
from hyperzero.game.state import GameState
from hyperzero.search.puct import (
    PolicyValueEvaluator,
    PUCTConfig,
    PUCTLeaf,
    PUCTSearchSession,
    run_puct,
)


@dataclass(frozen=True, slots=True)
class SelfPlayExample:
    """One training target from a self-play position."""

    board: np.ndarray
    policy: np.ndarray
    value: float
    legal_mask: np.ndarray
    player_to_move: int
    ply: int


@dataclass(frozen=True, slots=True)
class SelfPlayGame:
    """Self-play game output with finalized examples."""

    examples: tuple[SelfPlayExample, ...]
    actions: tuple[int, ...]
    winner: int | None
    terminal: bool


@dataclass(frozen=True, slots=True)
class _PendingExample:
    board: np.ndarray
    policy: np.ndarray
    legal_mask: np.ndarray
    player_to_move: int
    ply: int


def generate_game(
    config: GameConfig,
    evaluator: PolicyValueEvaluator,
    *,
    search_config: PUCTConfig | None = None,
    rng: np.random.Generator | None = None,
    use_line_counts: bool = False,
) -> SelfPlayGame:
    """Generate one game of self-play and return policy-value targets."""
    rng = np.random.default_rng() if rng is None else rng
    search_config = PUCTConfig() if search_config is None else search_config
    state = GameState.new(config, use_line_counts=use_line_counts)
    pending: list[_PendingExample] = []
    actions: list[int] = []

    while not state.terminal:
        legal_mask = state.legal_mask()
        result = run_puct(state, evaluator, search_config, rng=rng)
        pending.append(
            _PendingExample(
                board=state.canonical_board(flat=True).astype(np.float32),
                policy=result.policy.astype(np.float32),
                legal_mask=legal_mask.astype(bool),
                player_to_move=state.player_to_move,
                ply=state.ply,
            )
        )
        action = sample_action(result.policy, legal_mask, rng=rng)
        state.make_move(action)
        actions.append(action)

    examples = tuple(
        SelfPlayExample(
            board=example.board,
            policy=example.policy,
            value=float(state.outcome_for_player(example.player_to_move)),
            legal_mask=example.legal_mask,
            player_to_move=example.player_to_move,
            ply=example.ply,
        )
        for example in pending
    )
    return SelfPlayGame(
        examples=examples,
        actions=tuple(actions),
        winner=state.winner,
        terminal=state.terminal,
    )


@dataclass(slots=True)
class _ActiveSelfPlayGame:
    config: GameConfig
    rng: np.random.Generator
    use_line_counts: bool
    state: GameState = field(init=False)
    pending: list[_PendingExample] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.state = GameState.new(self.config, use_line_counts=self.use_line_counts)

    @property
    def terminal(self) -> bool:
        return self.state.terminal

    def build_search_session(self, search_config: PUCTConfig) -> PUCTSearchSession:
        """Create a fresh PUCT session for the current move."""
        return PUCTSearchSession(
            self.state,
            search_config,
            self.rng,
        )

    def apply_search_result(self, result_policy: np.ndarray) -> None:
        """Store a training target and advance one move."""
        legal_mask = self.state.legal_mask()
        self.pending.append(
            _PendingExample(
                board=self.state.canonical_board(flat=True).astype(np.float32),
                policy=result_policy.astype(np.float32),
                legal_mask=legal_mask.astype(bool),
                player_to_move=self.state.player_to_move,
                ply=self.state.ply,
            )
        )
        action = sample_action(result_policy, legal_mask, rng=self.rng)
        self.state.make_move(action)
        self.actions.append(action)

    def finalize(self) -> SelfPlayGame:
        """Return finalized examples with terminal outcomes."""
        examples = tuple(
            SelfPlayExample(
                board=example.board,
                policy=example.policy,
                value=float(self.state.outcome_for_player(example.player_to_move)),
                legal_mask=example.legal_mask,
                player_to_move=example.player_to_move,
                ply=example.ply,
            )
            for example in self.pending
        )
        return SelfPlayGame(
            examples=examples,
            actions=tuple(self.actions),
            winner=self.state.winner,
            terminal=self.state.terminal,
        )


def generate_games_batched(
    config: GameConfig,
    evaluator: PolicyValueEvaluator,
    *,
    games: int,
    search_config: PUCTConfig | None = None,
    rng: np.random.Generator | None = None,
    max_active_games: int | None = None,
    use_line_counts: bool = False,
) -> tuple[SelfPlayGame, ...]:
    """Generate self-play games while batching neural leaf evaluations."""
    if games <= 0:
        raise ValueError("games must be positive")
    rng = np.random.default_rng() if rng is None else rng
    search_config = PUCTConfig() if search_config is None else search_config
    max_active_games = games if max_active_games is None else max_active_games
    if max_active_games <= 0:
        raise ValueError("max_active_games must be positive")

    completed: list[SelfPlayGame] = []
    active: list[_ActiveSelfPlayGame] = []
    next_game_index = 0

    def start_games() -> None:
        nonlocal next_game_index
        while len(active) < max_active_games and next_game_index < games:
            active.append(
                _ActiveSelfPlayGame(
                    config,
                    np.random.default_rng(int(rng.integers(2**63 - 1))),
                    use_line_counts,
                )
            )
            next_game_index += 1

    start_games()
    while active:
        sessions: list[tuple[_ActiveSelfPlayGame, PUCTSearchSession]] = []
        for game in active:
            if not game.terminal:
                session = game.build_search_session(search_config)
                sessions.append((game, session))
        _initialize_search_sessions_batched(
            [session for _, session in sessions],
            evaluator,
        )

        while sessions:
            leaves_by_session = [
                (session, session.select_leaf())
                for _, session in sessions
                if not session.complete
            ]
            if not leaves_by_session:
                break
            _complete_leaves_batched(leaves_by_session, evaluator)

        still_active: list[_ActiveSelfPlayGame] = []
        for game, session in sessions:
            game.apply_search_result(session.result().policy)
            if game.terminal:
                completed.append(game.finalize())
            else:
                still_active.append(game)
        active = still_active
        start_games()

    return tuple(completed)


def _initialize_search_sessions_batched(
    sessions: list[PUCTSearchSession],
    evaluator: PolicyValueEvaluator,
) -> None:
    leaves = [session.select_leaf() for session in sessions]
    evaluations = evaluator.evaluate_many([leaf.state for leaf in leaves])
    for session, leaf, evaluation in zip(sessions, leaves, evaluations, strict=True):
        session.complete_leaf(leaf, evaluation)


def _complete_leaves_batched(
    leaves_by_session: list[tuple[PUCTSearchSession, PUCTLeaf]],
    evaluator: PolicyValueEvaluator,
) -> None:
    eval_items = [
        (session, leaf)
        for session, leaf in leaves_by_session
        if leaf.requires_evaluation
    ]
    evaluations = (
        evaluator.evaluate_many([leaf.state for _, leaf in eval_items])
        if eval_items
        else []
    )
    eval_index = 0
    for session, leaf in leaves_by_session:
        if leaf.requires_evaluation:
            session.complete_leaf(leaf, evaluations[eval_index])
            eval_index += 1
        else:
            session.complete_leaf(leaf)

"""Serializable game replay records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hyperzero.game.config import GameConfig
from hyperzero.game.state import GameState


@dataclass(frozen=True, slots=True)
class GameReplay:
    """A compact replay made from a config and flat action sequence."""

    config: GameConfig
    actions: tuple[int, ...]
    winner: int | None
    terminal: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_state(
        cls,
        state: GameState,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "GameReplay":
        """Build a replay from a state that still has move history."""
        return cls(
            config=state.config,
            actions=state.action_history(),
            winner=state.winner,
            terminal=state.terminal,
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def from_actions(
        cls,
        config: GameConfig,
        actions: tuple[int, ...] | list[int],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "GameReplay":
        """Play actions and return a replay with the resulting outcome."""
        state = GameState.new(config)
        for action in actions:
            state.make_move(action)
        return cls.from_state(state, metadata=metadata)

    def playback(self) -> GameState:
        """Replay actions into a fresh state and return the final state."""
        state = GameState.new(self.config)
        for action in self.actions:
            state.make_move(action)
        return state

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable replay dictionary."""
        return {
            "config": self.config.to_dict(),
            "actions": list(self.actions),
            "winner": self.winner,
            "terminal": self.terminal,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameReplay":
        """Build a replay from a dictionary produced by to_dict."""
        return cls(
            config=GameConfig.from_dict(data["config"]),
            actions=tuple(int(action) for action in data["actions"]),
            winner=data.get("winner"),
            terminal=bool(data.get("terminal", False)),
            metadata=dict(data.get("metadata", {})),
        )

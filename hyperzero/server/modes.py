"""Supported game modes for the local playable demo."""

from __future__ import annotations

from dataclasses import dataclass

from hyperzero.game import GameConfig

# Mirror the universal encoder defaults (UniversalEncoderConfig in
# hyperzero/universal/encoding.py). Configs beyond these limits cannot be
# encoded by the trained model, so reject them before creating a game.
MAX_RANK = 4
MAX_BOARD_EXTENT = 8


@dataclass(frozen=True, slots=True)
class ModeSpec:
    """User-facing metadata for one demo mode."""

    id: str
    label: str
    short_label: str
    description: str
    game_config: GameConfig

    @property
    def dimensions(self) -> int:
        return len(self.game_config.shape)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable metadata."""
        return {
            "id": self.id,
            "label": self.label,
            "short_label": self.short_label,
            "description": self.description,
            "dimensions": self.dimensions,
            "shape": list(self.game_config.shape),
            "connect_k": self.game_config.connect_k,
            "gravity_axis": self.game_config.gravity_axis,
            "action_shape": list(self.game_config.action_shape),
            "num_actions": self.game_config.num_actions,
        }


DEMO_MODES: dict[str, ModeSpec] = {
    "2d_6x7_k4": ModeSpec(
        id="2d_6x7_k4",
        label="2D Connect Four",
        short_label="2D",
        description="Classic 6x7 board with gravity.",
        game_config=GameConfig(shape=(6, 7), connect_k=4, gravity_axis=0),
    ),
    "3d_4x4x4_k4": ModeSpec(
        id="3d_4x4x4_k4",
        label="3D Connect Four",
        short_label="3D",
        description="4x4x4 board; choose a column in the 4x4 floor.",
        game_config=GameConfig(shape=(4, 4, 4), connect_k=4, gravity_axis=0),
    ),
    "4d_4x4x4x4_k4": ModeSpec(
        id="4d_4x4x4x4_k4",
        label="4D Connect Four",
        short_label="4D",
        description="4D board shown as selectable 3D slices.",
        game_config=GameConfig(shape=(4, 4, 4, 4), connect_k=4, gravity_axis=0),
    ),
}


def get_mode(mode_id: str) -> ModeSpec:
    """Return a mode by id, raising ValueError for unknown ids."""
    try:
        return DEMO_MODES[mode_id]
    except KeyError as exc:
        known = ", ".join(sorted(DEMO_MODES))
        raise ValueError(f"unknown mode {mode_id!r}; expected one of {known}") from exc


def build_mode_spec(
    shape: tuple[int, ...],
    connect_k: int,
    gravity_axis: int = 0,
) -> ModeSpec:
    """Build an ad-hoc mode from an arbitrary shape and connect-k.

    Validates against the universal encoder limits, then defers the remaining
    geometry checks (connect_k <= max(shape), positivity, gravity-axis range)
    to GameConfig. Raises ValueError on any invalid configuration.
    """
    shape = tuple(int(size) for size in shape)
    connect_k = int(connect_k)
    rank = len(shape)
    if not 1 <= rank <= MAX_RANK:
        raise ValueError(f"board rank {rank} must be between 1 and {MAX_RANK}")
    if shape and max(shape) > MAX_BOARD_EXTENT:
        raise ValueError(
            f"board extent {max(shape)} exceeds max_board_extent {MAX_BOARD_EXTENT}"
        )
    if connect_k > MAX_BOARD_EXTENT:
        raise ValueError(
            f"connect_k {connect_k} exceeds max_board_extent {MAX_BOARD_EXTENT}"
        )

    game_config = GameConfig(
        shape=shape,
        connect_k=connect_k,
        gravity_axis=gravity_axis,
    )
    extent_label = "×".join(str(size) for size in shape)
    return ModeSpec(
        id=f"{rank}d_{'x'.join(str(size) for size in shape)}_k{connect_k}",
        label=f"{rank}D {extent_label} (k={connect_k})",
        short_label=f"{rank}D",
        description=f"Custom {rank}D board {extent_label} with connect-{connect_k}.",
        game_config=game_config,
    )

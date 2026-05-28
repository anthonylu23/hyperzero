"""Shape-conditioned encodings for universal Connect-K models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from hyperzero.game.config import GameConfig
from hyperzero.game.state import GameState


@dataclass(frozen=True, slots=True)
class UniversalEncoderConfig:
    """Fixed encoder limits shared by one universal checkpoint."""

    max_rank: int = 4
    max_board_extent: int = 8
    line_features: bool = False

    def __post_init__(self) -> None:
        if self.max_rank <= 0:
            raise ValueError("max_rank must be positive")
        if self.max_board_extent <= 0:
            raise ValueError("max_board_extent must be positive")

    @property
    def feature_size(self) -> int:
        """Return the per-token feature width."""
        size = 4 * self.max_rank + 4
        if self.line_features:
            size += 4
        return size

    def to_dict(self) -> dict[str, int]:
        """Return a JSON/checkpoint serializable representation."""
        return {
            "max_rank": self.max_rank,
            "max_board_extent": self.max_board_extent,
            "line_features": self.line_features,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> UniversalEncoderConfig:
        """Build an encoder config from serialized data."""
        return cls(
            max_rank=int(data.get("max_rank", 4)),
            max_board_extent=int(data.get("max_board_extent", 8)),
            line_features=bool(data.get("line_features", False)),
        )


@dataclass(frozen=True, slots=True)
class UniversalPosition:
    """One variable-shape position encoded for a universal model."""

    config: GameConfig
    cell_features: np.ndarray
    cell_mask: np.ndarray
    action_features: np.ndarray
    action_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class UniversalBatch:
    """Padded torch tensors for a mixed-variant minibatch."""

    cell_features: torch.Tensor
    cell_mask: torch.Tensor
    action_features: torch.Tensor
    action_mask: torch.Tensor


def encode_state(
    state: GameState,
    encoder_config: UniversalEncoderConfig | None = None,
) -> UniversalPosition:
    """Encode a live game state from the current player's perspective."""
    encoder_config = (
        UniversalEncoderConfig() if encoder_config is None else encoder_config
    )
    line_counts = None
    if encoder_config.line_features and state.line_counts is not None:
        own_index = 1 if state.player_to_move == 1 else 0
        opponent_index = 1 - own_index
        line_counts = (
            state.line_counts[own_index].astype(np.int16, copy=False),
            state.line_counts[opponent_index].astype(np.int16, copy=False),
        )
    return encode_position(
        state.config,
        board=state.canonical_board(flat=True),
        legal_mask=state.legal_mask(),
        ply=state.ply,
        encoder_config=encoder_config,
        line_counts=line_counts,
    )


def encode_position(
    config: GameConfig,
    *,
    board: np.ndarray,
    legal_mask: np.ndarray,
    ply: int,
    encoder_config: UniversalEncoderConfig | None = None,
    line_counts: tuple[np.ndarray, np.ndarray] | None = None,
) -> UniversalPosition:
    """Encode a stored position using its game config and legal-action mask."""
    encoder_config = (
        UniversalEncoderConfig() if encoder_config is None else encoder_config
    )
    _validate_config(config, encoder_config)
    board = np.asarray(board, dtype=np.float32).reshape(config.num_cells)
    legal_mask = np.asarray(legal_mask, dtype=bool).reshape(config.num_actions)
    line_features = (
        _line_feature_tables(config, board, legal_mask, line_counts=line_counts)
        if encoder_config.line_features
        else None
    )

    cell_features = np.stack(
        [
            _token_features(
                config,
                coord=config.cell_coord(cell_index),
                first_scalar=float(board[cell_index]),
                ply=ply,
                encoder_config=encoder_config,
                line_features=(
                    None if line_features is None else line_features[0][cell_index]
                ),
            )
            for cell_index in range(config.num_cells)
        ]
    ).astype(np.float32)

    action_features = np.stack(
        [
            _token_features(
                config,
                coord=_action_full_coord(config, action),
                first_scalar=_column_fill_fraction(config, board, action),
                ply=ply,
                encoder_config=encoder_config,
                line_features=(
                    None if line_features is None else line_features[1][action]
                ),
            )
            for action in range(config.num_actions)
        ]
    ).astype(np.float32)

    return UniversalPosition(
        config=config,
        cell_features=cell_features,
        cell_mask=np.ones(config.num_cells, dtype=bool),
        action_features=action_features,
        action_mask=legal_mask,
    )


def collate_positions(
    positions: list[UniversalPosition] | tuple[UniversalPosition, ...],
    *,
    device: str | torch.device = "cpu",
) -> UniversalBatch:
    """Pad variable-size positions into a single torch batch."""
    if not positions:
        raise ValueError("positions must be nonempty")

    feature_size = positions[0].cell_features.shape[1]
    if any(position.cell_features.shape[1] != feature_size for position in positions):
        raise ValueError("all positions must use the same feature size")
    if any(position.action_features.shape[1] != feature_size for position in positions):
        raise ValueError("cell and action feature sizes must match")

    batch_size = len(positions)
    max_cells = max(position.cell_features.shape[0] for position in positions)
    max_actions = max(position.action_features.shape[0] for position in positions)
    resolved_device = torch.device(device)

    cell_features = np.zeros((batch_size, max_cells, feature_size), dtype=np.float32)
    cell_mask = np.zeros((batch_size, max_cells), dtype=bool)
    action_features = np.zeros(
        (batch_size, max_actions, feature_size),
        dtype=np.float32,
    )
    action_mask = np.zeros((batch_size, max_actions), dtype=bool)

    for index, position in enumerate(positions):
        cell_count = position.cell_features.shape[0]
        action_count = position.action_features.shape[0]
        cell_features[index, :cell_count] = position.cell_features
        cell_mask[index, :cell_count] = position.cell_mask
        action_features[index, :action_count] = position.action_features
        action_mask[index, :action_count] = position.action_mask

    return UniversalBatch(
        cell_features=torch.as_tensor(
            cell_features,
            dtype=torch.float32,
            device=resolved_device,
        ),
        cell_mask=torch.as_tensor(cell_mask, dtype=torch.bool, device=resolved_device),
        action_features=torch.as_tensor(
            action_features,
            dtype=torch.float32,
            device=resolved_device,
        ),
        action_mask=torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=resolved_device,
        ),
    )


def _validate_config(
    config: GameConfig,
    encoder_config: UniversalEncoderConfig,
) -> None:
    if len(config.shape) > encoder_config.max_rank:
        raise ValueError(
            f"config rank {len(config.shape)} exceeds max_rank "
            f"{encoder_config.max_rank}"
        )
    if max(config.shape) > encoder_config.max_board_extent:
        raise ValueError(
            f"config extent {max(config.shape)} exceeds max_board_extent "
            f"{encoder_config.max_board_extent}"
        )
    if config.connect_k > encoder_config.max_board_extent:
        raise ValueError(
            f"connect_k {config.connect_k} exceeds max_board_extent "
            f"{encoder_config.max_board_extent}"
        )


def _token_features(
    config: GameConfig,
    *,
    coord: tuple[int, ...],
    first_scalar: float,
    ply: int,
    encoder_config: UniversalEncoderConfig,
    line_features: np.ndarray | None = None,
) -> np.ndarray:
    max_rank = encoder_config.max_rank
    features = np.zeros(encoder_config.feature_size, dtype=np.float32)
    offset = 0
    features[offset] = float(first_scalar)
    offset += 1

    for axis in range(max_rank):
        if axis < len(config.shape):
            features[offset + axis] = _normalize_coord(coord[axis], config.shape[axis])
    offset += max_rank

    features[offset : offset + len(config.shape)] = 1.0
    offset += max_rank

    features[offset + config.gravity_axis] = 1.0
    offset += max_rank

    for axis, size in enumerate(config.shape):
        features[offset + axis] = float(size / encoder_config.max_board_extent)
    offset += max_rank

    features[offset] = float(config.connect_k / encoder_config.max_board_extent)
    features[offset + 1] = float(len(config.shape) / max_rank)
    features[offset + 2] = float(ply / max(1, config.num_cells))
    offset += 3

    if encoder_config.line_features:
        if line_features is None:
            line_features = np.zeros(4, dtype=np.float32)
        features[offset : offset + 4] = line_features
    return features


def _normalize_coord(value: int, size: int) -> float:
    if size <= 1:
        return 0.0
    return float((2.0 * value / (size - 1)) - 1.0)


def _action_full_coord(config: GameConfig, action: int) -> tuple[int, ...]:
    coord = [0] * len(config.shape)
    for axis, value in zip(
        config.action_axes,
        config.column_coord(action),
        strict=True,
    ):
        coord[axis] = value
    return tuple(coord)


def _column_fill_fraction(config: GameConfig, board: np.ndarray, action: int) -> float:
    occupied = np.count_nonzero(board[config.column_cells[action]] != 0)
    return float(occupied / config.gravity_size)


def _line_feature_tables(
    config: GameConfig,
    board: np.ndarray,
    legal_mask: np.ndarray,
    *,
    line_counts: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    lines = config.winning_lines
    if line_counts is None:
        line_values = board[lines]
        own_counts = (line_values > 0.0).sum(axis=1)
        opponent_counts = (line_values < 0.0).sum(axis=1)
    else:
        own_counts, opponent_counts = line_counts
    weights = np.asarray(
        [0.0, *[float(10**count) for count in range(1, config.connect_k + 1)]],
        dtype=np.float32,
    )

    open_own = np.where(opponent_counts == 0, weights[own_counts], 0.0)
    open_opponent = np.where(own_counts == 0, weights[opponent_counts], 0.0)

    line_cell = _line_cell_matrix(config)
    own_cell_scores = open_own @ line_cell
    opponent_cell_scores = open_opponent @ line_cell
    own_cell_strength = np.zeros(config.num_cells, dtype=np.float32)
    opponent_cell_strength = np.zeros(config.num_cells, dtype=np.float32)
    repeated_cells = lines.reshape(-1)
    np.maximum.at(
        own_cell_strength,
        repeated_cells,
        np.repeat(
            np.where(opponent_counts == 0, own_counts / config.connect_k, 0.0),
            config.connect_k,
        ),
    )
    np.maximum.at(
        opponent_cell_strength,
        repeated_cells,
        np.repeat(
            np.where(own_counts == 0, opponent_counts / config.connect_k, 0.0),
            config.connect_k,
        ),
    )
    cell_features = np.stack(
        [
            np.log1p(own_cell_scores),
            np.log1p(opponent_cell_scores),
            own_cell_strength,
            opponent_cell_strength,
        ],
        axis=1,
    ).astype(np.float32)

    action_features = np.zeros((config.num_actions, 4), dtype=np.float32)
    for action in range(config.num_actions):
        if not legal_mask[action]:
            continue
        height = int(np.count_nonzero(board[config.column_cells[action]] != 0))
        if height >= config.gravity_size:
            continue
        cell = int(config.column_cells[action, height])
        line_ids = np.asarray(config.lines_by_cell[cell], dtype=np.int32)
        if line_ids.size == 0:
            continue
        action_own_counts = own_counts[line_ids]
        action_opponent_counts = opponent_counts[line_ids]
        open_for_own = action_opponent_counts == 0
        open_for_block = action_own_counts == 0
        own_after = np.minimum(action_own_counts + 1, config.connect_k)
        own_scores = np.where(open_for_own, weights[own_after], 0.0)
        opponent_scores = np.where(
            open_for_block,
            weights[action_opponent_counts],
            0.0,
        )
        own_strength = np.where(
            open_for_own,
            own_after / config.connect_k,
            0.0,
        )
        opponent_strength = np.where(
            open_for_block,
            action_opponent_counts / config.connect_k,
            0.0,
        )
        action_features[action] = np.asarray(
            [
                float(np.log1p(own_scores.sum())),
                float(np.log1p(opponent_scores.sum())),
                float(own_strength.max(initial=0.0)),
                float(opponent_strength.max(initial=0.0)),
            ],
            dtype=np.float32,
        )

    return cell_features, action_features


def _line_cell_matrix(config: GameConfig) -> np.ndarray:
    return _line_cell_matrix_cached(
        tuple(config.shape),
        config.connect_k,
        config.gravity_axis,
    )


@lru_cache(maxsize=32)
def _line_cell_matrix_cached(
    shape: tuple[int, ...],
    connect_k: int,
    gravity_axis: int,
) -> np.ndarray:
    config = GameConfig(
        shape=shape,
        connect_k=connect_k,
        gravity_axis=gravity_axis,
    )
    line_cell = np.zeros(
        (len(config.winning_lines), config.num_cells),
        dtype=np.float32,
    )
    rows = np.repeat(np.arange(len(config.winning_lines)), config.connect_k)
    line_cell[rows, config.winning_lines.reshape(-1)] = 1.0
    return line_cell

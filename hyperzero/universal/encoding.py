"""Shape-conditioned encodings for universal Connect-K models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hyperzero.game.config import GameConfig
from hyperzero.game.state import GameState


@dataclass(frozen=True, slots=True)
class UniversalEncoderConfig:
    """Fixed encoder limits shared by one universal checkpoint."""

    max_rank: int = 4
    max_board_extent: int = 8

    def __post_init__(self) -> None:
        if self.max_rank <= 0:
            raise ValueError("max_rank must be positive")
        if self.max_board_extent <= 0:
            raise ValueError("max_board_extent must be positive")

    @property
    def feature_size(self) -> int:
        """Return the per-token feature width."""
        return 4 * self.max_rank + 4

    def to_dict(self) -> dict[str, int]:
        """Return a JSON/checkpoint serializable representation."""
        return {
            "max_rank": self.max_rank,
            "max_board_extent": self.max_board_extent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> UniversalEncoderConfig:
        """Build an encoder config from serialized data."""
        return cls(
            max_rank=int(data.get("max_rank", 4)),
            max_board_extent=int(data.get("max_board_extent", 8)),
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
    return encode_position(
        state.config,
        board=state.canonical_board(flat=True),
        legal_mask=state.legal_mask(),
        ply=state.ply,
        encoder_config=encoder_config,
    )


def encode_position(
    config: GameConfig,
    *,
    board: np.ndarray,
    legal_mask: np.ndarray,
    ply: int,
    encoder_config: UniversalEncoderConfig | None = None,
) -> UniversalPosition:
    """Encode a stored position using its game config and legal-action mask."""
    encoder_config = (
        UniversalEncoderConfig() if encoder_config is None else encoder_config
    )
    _validate_config(config, encoder_config)
    board = np.asarray(board, dtype=np.float32).reshape(config.num_cells)
    legal_mask = np.asarray(legal_mask, dtype=bool).reshape(config.num_actions)

    cell_features = np.stack(
        [
            _token_features(
                config,
                coord=config.cell_coord(cell_index),
                first_scalar=float(board[cell_index]),
                ply=ply,
                encoder_config=encoder_config,
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

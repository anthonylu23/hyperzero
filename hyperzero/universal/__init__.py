"""Universal multi-variant agent scaffolding."""

from hyperzero.universal.encoding import (
    UniversalBatch,
    UniversalEncoderConfig,
    UniversalPosition,
    collate_positions,
    encode_position,
    encode_state,
)

__all__ = [
    "UniversalBatch",
    "UniversalEncoderConfig",
    "UniversalPosition",
    "collate_positions",
    "encode_position",
    "encode_state",
]

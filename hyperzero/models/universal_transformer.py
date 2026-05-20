"""Universal token transformer for mixed Connect-K variants."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hyperzero.universal.encoding import UniversalBatch, UniversalEncoderConfig


@dataclass(frozen=True, slots=True)
class UniversalModelConfig:
    """Architecture and encoder settings for one universal checkpoint."""

    encoder: UniversalEncoderConfig = UniversalEncoderConfig()
    hidden_size: int = 128
    residual_blocks: int = 2
    heads: int = 4

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.residual_blocks <= 0:
            raise ValueError("residual_blocks must be positive")
        if self.heads <= 0:
            raise ValueError("heads must be positive")

    def to_dict(self) -> dict[str, object]:
        """Return a checkpoint-serializable representation."""
        return {
            "encoder": self.encoder.to_dict(),
            "hidden_size": self.hidden_size,
            "residual_blocks": self.residual_blocks,
            "heads": self.heads,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> UniversalModelConfig:
        """Build model config from a checkpoint payload."""
        encoder_payload = data.get("encoder", {})
        if not isinstance(encoder_payload, dict):
            raise ValueError("universal model encoder config must be a mapping")
        return cls(
            encoder=UniversalEncoderConfig.from_dict(encoder_payload),
            hidden_size=int(data.get("hidden_size", 128)),
            residual_blocks=int(data.get("residual_blocks", 2)),
            heads=int(data.get("heads", 4)),
        )


class UniversalPolicyValueTransformer(nn.Module):
    """Score variable action spaces from shared cell/action token context."""

    def __init__(self, config: UniversalModelConfig | None = None) -> None:
        super().__init__()
        self.config = UniversalModelConfig() if config is None else config
        heads = self.config.heads
        if self.config.hidden_size % heads != 0:
            heads = 1

        feature_size = self.config.encoder.feature_size
        hidden_size = self.config.hidden_size
        self.global_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.cell_projection = nn.Linear(feature_size, hidden_size)
        self.action_projection = nn.Linear(feature_size, hidden_size)
        self.cell_type = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.action_type = nn.Parameter(torch.zeros(1, 1, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.config.residual_blocks,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

    def forward(self, batch: UniversalBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded action logits and values for a universal batch."""
        cell_features = batch.cell_features.to(dtype=torch.float32)
        action_features = batch.action_features.to(dtype=torch.float32)
        if cell_features.ndim != 3 or action_features.ndim != 3:
            raise ValueError("universal features must be batch x tokens x features")
        if cell_features.shape[0] != action_features.shape[0]:
            raise ValueError("cell and action batches must have the same batch size")
        if cell_features.shape[2] != self.config.encoder.feature_size:
            raise ValueError(
                f"cell feature size {cell_features.shape[2]} does not match "
                f"{self.config.encoder.feature_size}"
            )
        if action_features.shape[2] != self.config.encoder.feature_size:
            raise ValueError(
                f"action feature size {action_features.shape[2]} does not match "
                f"{self.config.encoder.feature_size}"
            )

        batch_size = cell_features.shape[0]
        global_tokens = self.global_token.expand(batch_size, -1, -1)
        cell_tokens = self.cell_projection(cell_features) + self.cell_type
        action_tokens = self.action_projection(action_features) + self.action_type
        tokens = torch.cat([global_tokens, cell_tokens, action_tokens], dim=1)

        global_mask = torch.ones(
            (batch_size, 1),
            dtype=torch.bool,
            device=tokens.device,
        )
        token_mask = torch.cat(
            [global_mask, batch.cell_mask, batch.action_mask],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=~token_mask)
        global_context = encoded[:, 0]
        action_start = 1 + cell_tokens.shape[1]
        encoded_actions = encoded[:, action_start:]
        expanded_context = global_context.unsqueeze(1).expand(
            -1,
            encoded_actions.shape[1],
            -1,
        )
        policy_logits = self.policy_head(
            torch.cat([encoded_actions, expanded_context], dim=-1)
        ).squeeze(-1)
        value = self.value_head(global_context).squeeze(-1)
        return policy_logits, value

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
    input_layer_norm: bool = False
    rank_adapters: bool = False
    rank_head_adapters: bool = False
    adapter_size: int = 0
    line_policy_residual: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.residual_blocks <= 0:
            raise ValueError("residual_blocks must be positive")
        if self.heads <= 0:
            raise ValueError("heads must be positive")
        if self.adapter_size < 0:
            raise ValueError("adapter_size must be nonnegative")
        if self.line_policy_residual and not self.encoder.line_features:
            raise ValueError("line_policy_residual requires encoder line_features")

    def to_dict(self) -> dict[str, object]:
        """Return a checkpoint-serializable representation."""
        return {
            "encoder": self.encoder.to_dict(),
            "hidden_size": self.hidden_size,
            "residual_blocks": self.residual_blocks,
            "heads": self.heads,
            "input_layer_norm": self.input_layer_norm,
            "rank_adapters": self.rank_adapters,
            "rank_head_adapters": self.rank_head_adapters,
            "adapter_size": self.adapter_size,
            "line_policy_residual": self.line_policy_residual,
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
            input_layer_norm=bool(data.get("input_layer_norm", False)),
            rank_adapters=bool(data.get("rank_adapters", False)),
            rank_head_adapters=bool(data.get("rank_head_adapters", False)),
            adapter_size=int(data.get("adapter_size", 0)),
            line_policy_residual=bool(data.get("line_policy_residual", False)),
        )


class _RankAdapter(nn.Module):
    """Small rank-specific residual branch for universal token states."""

    def __init__(self, hidden_size: int, adapter_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, adapter_size),
            nn.GELU(),
            nn.Linear(adapter_size, hidden_size),
        )
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


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
        self.input_norm = (
            nn.LayerNorm(hidden_size) if self.config.input_layer_norm else nn.Identity()
        )
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
        adapter_size = self.config.adapter_size or max(8, hidden_size // 4)
        self.rank_adapter_layers = (
            nn.ModuleList(
                [
                    _RankAdapter(hidden_size, adapter_size)
                    for _ in range(self.config.encoder.max_rank)
                ]
            )
            if self.config.rank_adapters
            else None
        )
        self.policy_rank_head_adapters = (
            nn.ModuleList(
                [
                    _RankAdapter(hidden_size, adapter_size)
                    for _ in range(self.config.encoder.max_rank)
                ]
            )
            if self.config.rank_head_adapters
            else None
        )
        self.value_rank_head_adapters = (
            nn.ModuleList(
                [
                    _RankAdapter(hidden_size, adapter_size)
                    for _ in range(self.config.encoder.max_rank)
                ]
            )
            if self.config.rank_head_adapters
            else None
        )
        self.line_policy_residual = (
            nn.Sequential(
                nn.LayerNorm(4),
                nn.Linear(4, adapter_size),
                nn.GELU(),
                nn.Linear(adapter_size, 1),
            )
            if self.config.line_policy_residual
            else None
        )
        if self.line_policy_residual is not None:
            last = self.line_policy_residual[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
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
        tokens = self.input_norm(
            torch.cat([global_tokens, cell_tokens, action_tokens], dim=1)
        )

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
        if self.rank_adapter_layers is not None:
            encoded = encoded + self._rank_adapter_delta(encoded, cell_features)
        global_context = encoded[:, 0]
        action_start = 1 + cell_tokens.shape[1]
        encoded_actions = encoded[:, action_start:]
        rank_indices = self._rank_indices(cell_features)
        if self.policy_rank_head_adapters is not None:
            encoded_actions = encoded_actions + self._rank_head_delta(
                encoded_actions,
                rank_indices,
                self.policy_rank_head_adapters,
            )
        if self.value_rank_head_adapters is not None:
            global_context = global_context + self._rank_head_delta(
                global_context,
                rank_indices,
                self.value_rank_head_adapters,
            )
        expanded_context = global_context.unsqueeze(1).expand(
            -1,
            encoded_actions.shape[1],
            -1,
        )
        policy_logits = self.policy_head(
            torch.cat([encoded_actions, expanded_context], dim=-1)
        ).squeeze(-1)
        if self.line_policy_residual is not None:
            line_features = action_features[..., -4:]
            policy_logits = policy_logits + self.line_policy_residual(
                line_features
            ).squeeze(-1)
        value = self.value_head(global_context).squeeze(-1)
        return policy_logits, value

    def _rank_adapter_delta(
        self,
        encoded: torch.Tensor,
        cell_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.rank_adapter_layers is None:
            return torch.zeros_like(encoded)
        rank_indices = self._rank_indices(cell_features)
        delta = torch.zeros_like(encoded)
        for rank_index, adapter in enumerate(self.rank_adapter_layers):
            rows = torch.nonzero(rank_indices == rank_index, as_tuple=True)[0]
            if rows.numel() > 0:
                delta.index_copy_(0, rows, adapter(encoded.index_select(0, rows)))
        return delta

    def _rank_head_delta(
        self,
        tokens: torch.Tensor,
        rank_indices: torch.Tensor,
        adapters: nn.ModuleList,
    ) -> torch.Tensor:
        delta = torch.zeros_like(tokens)
        for rank_index, adapter in enumerate(adapters):
            rows = torch.nonzero(rank_indices == rank_index, as_tuple=True)[0]
            if rows.numel() > 0:
                delta.index_copy_(0, rows, adapter(tokens.index_select(0, rows)))
        return delta

    def _rank_indices(self, cell_features: torch.Tensor) -> torch.Tensor:
        max_rank = self.config.encoder.max_rank
        rank_offset = 1 + max_rank
        rank_presence = cell_features[:, :, rank_offset : rank_offset + max_rank].amax(
            dim=1
        )
        return (
            rank_presence.sum(dim=1).round().to(dtype=torch.long).clamp(1, max_rank) - 1
        )

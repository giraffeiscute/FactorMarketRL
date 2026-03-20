"""Portfolio attention model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .config import ModelConfig


class PortfolioAttentionModel(nn.Module):
    """Minimal portfolio model with separate stock and market temporal branches.

    Time position is applied as `x_t = w_t + p_t`.
    Stock identity position is applied as `x_{s,t} = [z_{s,t}; e_s]`.
    """

    def __init__(self, config: ModelConfig, *, num_stocks: int) -> None:
        super().__init__()
        if num_stocks <= 0:
            raise ValueError("num_stocks must be positive before model construction.")
        attention_input_dim = config.cross_sectional_dim + config.stock_id_embedding_dim
        if attention_input_dim % config.attention_heads != 0:
            raise ValueError("attention_input_dim must be divisible by attention_heads.")

        self.config = config
        self.num_stocks = num_stocks
        self.time_position_mode = "add"
        self.id_position_mode = "concat"

        self.stock_input_proj = nn.Linear(config.stock_feature_dim, config.stock_temporal_dim)
        self.market_input_proj = nn.Linear(config.market_feature_dim, config.market_temporal_dim)
        self.stock_time_position = nn.Embedding(config.lookback, config.stock_temporal_dim)
        self.market_time_position = nn.Embedding(config.lookback, config.market_temporal_dim)

        stock_temporal_heads = 2 if config.stock_temporal_dim % 2 == 0 else 1
        market_temporal_heads = 2 if config.market_temporal_dim % 2 == 0 else 1

        stock_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.stock_temporal_dim,
            nhead=stock_temporal_heads,
            dim_feedforward=max(32, config.stock_temporal_dim * 2),
            dropout=config.dropout,
            batch_first=True,
        )
        market_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.market_temporal_dim,
            nhead=market_temporal_heads,
            dim_feedforward=max(32, config.market_temporal_dim * 2),
            dropout=config.dropout,
            batch_first=True,
        )
        self.stock_temporal_encoder = nn.TransformerEncoder(stock_encoder_layer, num_layers=1)
        self.market_temporal_encoder = nn.TransformerEncoder(market_encoder_layer, num_layers=1)

        self.fusion_projection = nn.Linear(
            config.stock_temporal_dim + config.market_temporal_dim,
            config.cross_sectional_dim,
        )
        self.stock_id_embedding = nn.Embedding(num_stocks, config.stock_id_embedding_dim)
        self.stock_attention = nn.MultiheadAttention(
            embed_dim=attention_input_dim,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.stock_head = nn.Linear(attention_input_dim, 1)
        self.cash_head = nn.Linear(attention_input_dim, 1)
        self.post_attention_norm = nn.LayerNorm(attention_input_dim)

    def apply_time_position(self, temporal_content: torch.Tensor, branch: str) -> torch.Tensor:
        """Apply `x_t = w_t + p_t`."""

        if temporal_content.ndim != 3:
            raise ValueError("temporal_content must have shape [batch_like, lookback, hidden].")
        sequence_length = temporal_content.shape[1]
        positions = torch.arange(sequence_length, device=temporal_content.device)
        if branch == "stock":
            pos_embedding = self.stock_time_position(positions)
        elif branch == "market":
            pos_embedding = self.market_time_position(positions)
        else:
            raise ValueError(f"Unsupported branch: {branch}")
        return temporal_content + pos_embedding.unsqueeze(0)

    def append_stock_identity(
        self,
        stock_representation: torch.Tensor,
        stock_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Apply `x_{s,t} = [z_{s,t}; e_s]` in the cross-sectional feature dimension."""

        identity = self.stock_id_embedding(stock_indices)
        return torch.cat([stock_representation, identity], dim=-1)

    def forward(
        self,
        x_stock: torch.Tensor,
        x_market: torch.Tensor,
        stock_indices: torch.Tensor,
        target_returns: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if x_stock.ndim != 4:
            raise ValueError("x_stock must have shape [B, N, L, F_stock].")
        if x_market.ndim != 3:
            raise ValueError("x_market must have shape [B, L, 3].")
        if stock_indices.ndim != 2:
            raise ValueError("stock_indices must have shape [B, N].")

        batch_size, num_stocks, lookback, stock_feature_dim = x_stock.shape
        assert stock_feature_dim == self.config.stock_feature_dim
        assert x_market.shape == (batch_size, lookback, self.config.market_feature_dim)
        assert stock_indices.shape == (batch_size, num_stocks)
        if num_stocks > self.num_stocks:
            raise ValueError(
                f"Received batch with num_stocks={num_stocks}, but model was constructed for {self.num_stocks} stocks."
            )

        stock_flat = x_stock.reshape(batch_size * num_stocks, lookback, stock_feature_dim)
        stock_content = self.stock_input_proj(stock_flat)
        stock_sequence = self.apply_time_position(stock_content, branch="stock")
        stock_temporal = self.stock_temporal_encoder(stock_sequence)
        z_i = stock_temporal.mean(dim=1).reshape(batch_size, num_stocks, self.config.stock_temporal_dim)

        market_content = self.market_input_proj(x_market)
        market_sequence = self.apply_time_position(market_content, branch="market")
        market_temporal = self.market_temporal_encoder(market_sequence)
        market_summary = market_temporal.mean(dim=1)

        fused = torch.cat(
            [z_i, market_summary.unsqueeze(1).expand(-1, num_stocks, -1)],
            dim=-1,
        )
        projected = self.fusion_projection(fused)
        stock_tokens = self.append_stock_identity(projected, stock_indices)

        attention_out, attention_weights = self.stock_attention(
            stock_tokens,
            stock_tokens,
            stock_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        attention_out = self.post_attention_norm(attention_out + stock_tokens)

        stock_logits = self.stock_head(attention_out).squeeze(-1)
        pooled_stock = attention_out.mean(dim=1)
        cash_logit = self.cash_head(pooled_stock).squeeze(-1)
        allocation_logits = torch.cat([stock_logits, cash_logit.unsqueeze(-1)], dim=-1)
        allocation = torch.softmax(allocation_logits, dim=-1)
        stock_weights = allocation[:, :-1]
        cash_weight = allocation[:, -1]

        portfolio_return = None
        if target_returns is not None:
            if target_returns.shape != (batch_size, num_stocks):
                raise ValueError("target_returns must have shape [B, N].")
            portfolio_return = (stock_weights * target_returns).sum(dim=-1)

        debug_info = {
            "time_position_mode": self.time_position_mode,
            "id_position_mode": self.id_position_mode,
            "stock_attention_token_count": int(stock_tokens.shape[1]),
            "cash_token_in_attention": False,
            "stock_temporal_shape": tuple(stock_temporal.shape),
            "market_temporal_shape": tuple(market_temporal.shape),
            "fused_shape": tuple(fused.shape),
            "attention_weight_shape": tuple(attention_weights.shape),
        }

        return {
            "stock_weights": stock_weights,
            "cash_weight": cash_weight,
            "stock_logits": stock_logits,
            "cash_logit": cash_logit,
            "portfolio_return": portfolio_return,
            "debug_info": debug_info,
        }

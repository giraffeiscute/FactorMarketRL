"""Scenario-aware portfolio model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import math

from .config import ModelConfig


class PortfolioAttentionModel(nn.Module):
    """Portfolio model that preserves scenario and time structure.

    Expected tensor layout:
    - `x_stock`: [S, T, N, F_stock]
    - `x_market`: [S, T, F_market]
    - `stock_indices`: [S, N]
    - `target_returns`: [S, T, N]

    The forward pass keeps `S` (scenario) and `T` (time) separate and returns:
    - `stock_weights`: [S, T, N]
    - `cash_weight`: [S, T]
    - `portfolio_return`: [S, T]

    To avoid future leakage, the model uses only the current step and causal
    running summaries up to the current step. It does not flatten scenario and
    time into a single path.
    """

    def __init__(self, config: ModelConfig, *, num_stocks: int, max_lookback: int) -> None:
        super().__init__()
        if num_stocks <= 0:
            raise ValueError("num_stocks must be positive before model construction.")
        if max_lookback <= 0:
            raise ValueError("max_lookback must be positive before model construction.")

        self.config = config
        self.num_stocks = num_stocks
        self.max_lookback = max_lookback
        self.time_position_mode = config.time_positional_encoding_type
        self.id_position_mode = "concat"

        self.stock_input_proj = nn.Linear(config.stock_feature_dim, config.cross_sectional_dim)
        self.market_input_proj = nn.Linear(config.market_feature_dim, config.market_temporal_dim)
        self.stock_id_embedding = nn.Embedding(num_stocks, config.stock_id_embedding_dim)

        stock_feature_width = (
            config.cross_sectional_dim * 2
            + config.market_temporal_dim * 2
            + config.stock_id_embedding_dim
        )
        cash_feature_width = (
            config.cross_sectional_dim * 2 + config.market_temporal_dim * 2
        )
        cash_hidden_dim = max(4, config.market_temporal_dim)

        self.stock_score = nn.Sequential(
            nn.Linear(stock_feature_width, config.cross_sectional_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.cross_sectional_dim, 1),
        )
        self.cash_score = nn.Sequential(
            nn.Linear(cash_feature_width, cash_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(cash_hidden_dim, 1),
        )

    @staticmethod
    def _causal_running_mean(values: torch.Tensor) -> torch.Tensor:
        if values.ndim < 2:
            raise ValueError("Expected at least 2 dimensions for causal running mean.")
        steps = torch.arange(
            1,
            values.shape[1] + 1,
            device=values.device,
            dtype=values.dtype,
        )
        view_shape = [1, values.shape[1]] + [1] * (values.ndim - 2)
        return values.cumsum(dim=1) / steps.view(*view_shape)

    @staticmethod
    def _build_sinusoidal_time_encoding(
        *,
        time_steps: int,
        embedding_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if time_steps <= 0:
            raise ValueError(f"time_steps must be positive, received {time_steps}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, received {embedding_dim}.")

        positions = torch.arange(time_steps, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / embedding_dim)
        )
        encoding = torch.zeros((time_steps, embedding_dim), device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0)

    def forward(
        self,
        x_stock: torch.Tensor,
        x_market: torch.Tensor,
        stock_indices: torch.Tensor,
        target_returns: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if x_stock.ndim != 4:
            raise ValueError("x_stock must have shape [S, T, N, F_stock].")
        if x_market.ndim != 3:
            raise ValueError("x_market must have shape [S, T, F_market].")
        if stock_indices.ndim != 2:
            raise ValueError("stock_indices must have shape [S, N].")

        num_scenarios, time_steps, num_stocks, stock_feature_dim = x_stock.shape
        assert stock_feature_dim == self.config.stock_feature_dim
        assert x_market.shape == (num_scenarios, time_steps, self.config.market_feature_dim)
        assert stock_indices.shape == (num_scenarios, num_stocks)
        if time_steps > self.max_lookback:
            raise ValueError(
                f"Received time_steps={time_steps}, but model was constructed for max_lookback={self.max_lookback}."
            )
        if num_stocks > self.num_stocks:
            raise ValueError(
                f"Received num_stocks={num_stocks}, but model was constructed for {self.num_stocks}."
            )

        stock_current = self.stock_input_proj(x_stock)
        market_current = self.market_input_proj(x_market)

        if self.time_position_mode == "sinusoidal":
            stock_time_encoding = self._build_sinusoidal_time_encoding(
                time_steps=time_steps,
                embedding_dim=self.config.cross_sectional_dim,
                device=stock_current.device,
                dtype=stock_current.dtype,
            ).unsqueeze(2)
            market_time_encoding = self._build_sinusoidal_time_encoding(
                time_steps=time_steps,
                embedding_dim=self.config.market_temporal_dim,
                device=market_current.device,
                dtype=market_current.dtype,
            )
            stock_current = stock_current + stock_time_encoding
            market_current = market_current + market_time_encoding

        stock_running = self._causal_running_mean(stock_current)
        market_running = self._causal_running_mean(market_current)

        market_current_expanded = market_current.unsqueeze(2).expand(-1, -1, num_stocks, -1)
        market_running_expanded = market_running.unsqueeze(2).expand(-1, -1, num_stocks, -1)
        stock_identity = self.stock_id_embedding(stock_indices).unsqueeze(1).expand(
            -1, time_steps, -1, -1
        )

        stock_features = torch.cat(
            [
                stock_current,
                stock_running,
                market_current_expanded,
                market_running_expanded,
                stock_identity,
            ],
            dim=-1,
        )
        stock_logits = self.stock_score(stock_features).squeeze(-1)

        pooled_stock_current = stock_current.mean(dim=2)
        pooled_stock_running = stock_running.mean(dim=2)
        cash_features = torch.cat(
            [pooled_stock_current, pooled_stock_running, market_current, market_running],
            dim=-1,
        )
        cash_logit = self.cash_score(cash_features).squeeze(-1)

        allocation_logits = torch.cat([stock_logits, cash_logit.unsqueeze(-1)], dim=-1)
        allocation = torch.softmax(allocation_logits, dim=-1)
        stock_weights = allocation[..., :-1]
        cash_weight = allocation[..., -1]

        portfolio_return = None
        if target_returns is not None:
            expected_shape = (num_scenarios, time_steps, num_stocks)
            if target_returns.shape != expected_shape:
                raise ValueError(
                    f"target_returns must have shape {expected_shape}, received {tuple(target_returns.shape)}."
                )
            portfolio_return = (stock_weights * target_returns).sum(dim=-1)

        debug_info = {
            "time_position_mode": self.time_position_mode,
            "id_position_mode": self.id_position_mode,
            "stock_current_shape": tuple(stock_current.shape),
            "stock_running_shape": tuple(stock_running.shape),
            "market_current_shape": tuple(market_current.shape),
            "market_running_shape": tuple(market_running.shape),
            "stock_feature_shape": tuple(stock_features.shape),
        }

        return {
            "stock_weights": stock_weights,
            "cash_weight": cash_weight,
            "stock_logits": stock_logits,
            "cash_logit": cash_logit,
            "portfolio_return": portfolio_return,
            "debug_info": debug_info,
        }

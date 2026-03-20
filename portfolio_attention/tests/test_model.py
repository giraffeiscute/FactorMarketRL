from __future__ import annotations

import torch

from portfolio_attention.config import ModelConfig
from portfolio_attention.model import PortfolioAttentionModel


def test_forward_shapes_and_weight_sum() -> None:
    config = ModelConfig()
    model = PortfolioAttentionModel(config, num_stocks=5)
    x_stock = torch.randn(2, 5, 60, 4)
    x_market = torch.randn(2, 60, 3)
    stock_indices = torch.arange(5).unsqueeze(0).repeat(2, 1)
    target_returns = torch.randn(2, 5)

    outputs = model(x_stock, x_market, stock_indices, target_returns)

    assert outputs["stock_weights"].shape == (2, 5)
    assert outputs["cash_weight"].shape == (2,)
    assert outputs["stock_logits"].shape == (2, 5)
    assert outputs["cash_logit"].shape == (2,)
    assert outputs["portfolio_return"].shape == (2,)
    total_weight = outputs["stock_weights"].sum(dim=-1) + outputs["cash_weight"]
    assert torch.allclose(total_weight, torch.ones_like(total_weight), atol=1e-6)


def test_cash_not_in_stock_attention_tokens() -> None:
    config = ModelConfig()
    model = PortfolioAttentionModel(config, num_stocks=4)
    x_stock = torch.randn(1, 4, 60, 4)
    x_market = torch.randn(1, 60, 3)
    stock_indices = torch.arange(4).unsqueeze(0)
    outputs = model(x_stock, x_market, stock_indices)

    assert outputs["debug_info"]["cash_token_in_attention"] is False
    assert outputs["debug_info"]["stock_attention_token_count"] == 4


def test_time_position_is_addition() -> None:
    config = ModelConfig()
    model = PortfolioAttentionModel(config, num_stocks=3)
    temporal_content = torch.randn(2, 60, config.stock_temporal_dim)
    positioned = model.apply_time_position(temporal_content, branch="stock")
    expected_delta = model.stock_time_position(torch.arange(60)).unsqueeze(0).expand_as(positioned)

    assert torch.allclose(positioned - temporal_content, expected_delta, atol=1e-6)


def test_id_position_is_concat() -> None:
    config = ModelConfig()
    model = PortfolioAttentionModel(config, num_stocks=3)
    representation = torch.randn(1, 3, config.cross_sectional_dim)
    stock_indices = torch.arange(3).unsqueeze(0)
    concatenated = model.append_stock_identity(representation, stock_indices)
    identity = model.stock_id_embedding(stock_indices)

    assert concatenated.shape[-1] == config.cross_sectional_dim + config.stock_id_embedding_dim
    assert torch.allclose(concatenated[..., : config.cross_sectional_dim], representation)
    assert torch.allclose(concatenated[..., config.cross_sectional_dim :], identity)

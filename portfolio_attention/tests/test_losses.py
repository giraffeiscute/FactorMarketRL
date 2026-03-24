from __future__ import annotations

import pytest
import torch

from portfolio_attention.losses import (
    return_loss,
    sharpe_loss,
    differential_sharpe_loss,
    sortino_loss,
    max_drawdown_loss,
    cvar_loss,
    build_loss
)


def test_return_loss_multiplicative() -> None:
    # 1.1 * 1.2 * 0.9 - 1 = 1.188 - 1 = 0.188
    # loss = -0.188
    returns = torch.tensor([0.1, 0.2, -0.1])
    loss = return_loss(returns, mode="multiplicative")
    assert torch.isclose(loss, torch.tensor(-0.188), atol=1e-5)


def test_return_loss_additive() -> None:
    # sum = 0.2
    # loss = -0.2
    returns = torch.tensor([0.1, 0.2, -0.1])
    loss = return_loss(returns, mode="additive")
    assert torch.isclose(loss, torch.tensor(-0.2), atol=1e-5)


def test_sharpe_loss_path() -> None:
    # Path [T]
    returns = torch.tensor([0.01, 0.03, -0.01, 0.02])
    loss = sharpe_loss(returns)
    assert torch.isfinite(loss)
    
    # Batch Path [B, T]
    returns_bt = torch.randn(2, 10)
    loss_bt = sharpe_loss(returns_bt)
    assert torch.isfinite(loss_bt)


def test_sharpe_loss_fallback() -> None:
    returns = torch.tensor([0.02])
    # Now warns about "time steps"
    with pytest.warns(RuntimeWarning, match="too few time steps"):
        loss = sharpe_loss(returns)
    # Fallback to return_loss (multiplicative)
    assert torch.isclose(loss, torch.tensor(-0.02), atol=1e-7)


def test_dsr_loss() -> None:
    returns = torch.randn(2, 10)
    loss = differential_sharpe_loss(returns)
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)


def test_sortino_loss() -> None:
    returns = torch.randn(2, 10)
    loss = sortino_loss(returns)
    assert torch.isfinite(loss)


def test_max_drawdown_loss() -> None:
    returns = torch.tensor([0.1, -0.2, 0.3])
    # multiplicative: equity = [1.1, 0.88, 1.144], peak = [1.1, 1.1, 1.144]
    # dd = [0, (1.1-0.88)/1.1, 0] = [0, 0.2, 0] -> max = 0.2
    loss = max_drawdown_loss(returns, mode="multiplicative")
    assert torch.isclose(loss, torch.tensor(0.2), atol=1e-5)
    assert loss >= 0


def test_cvar_loss() -> None:
    returns = torch.tensor([-0.1, -0.2, 0.1, 0.2, 0.0])
    # losses = [0.1, 0.2, -0.1, -0.2, 0.0]
    # alpha=0.2 (1/5) -> 1-alpha = 0.8 quantile -> 0.2
    # tail = [0.2] -> mean = 0.2
    loss = cvar_loss(returns, alpha=0.2)
    assert torch.isclose(loss, torch.tensor(0.2), atol=1e-5)
    assert loss >= 0


def test_build_loss_dispatch() -> None:
    returns = torch.randn(10)
    assert torch.isfinite(build_loss("SR", returns))
    assert torch.isfinite(build_loss("differential_sharpe", returns))
    assert torch.isfinite(build_loss("Max_Drawdown", returns))

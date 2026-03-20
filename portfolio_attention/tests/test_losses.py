from __future__ import annotations

import pytest
import torch

from portfolio_attention.losses import return_loss, sharpe_loss


def test_return_loss_runs() -> None:
    returns = torch.tensor([0.01, 0.02, -0.01])
    loss = return_loss(returns)
    assert torch.isclose(loss, torch.tensor(-0.006666667), atol=1e-7)


def test_sharpe_loss_on_normal_batch() -> None:
    returns = torch.tensor([0.01, 0.03, -0.01, 0.02])
    loss = sharpe_loss(returns)
    assert torch.isfinite(loss)


def test_sharpe_loss_falls_back_on_tiny_batch() -> None:
    returns = torch.tensor([0.02])
    with pytest.warns(RuntimeWarning, match="too few samples"):
        loss = sharpe_loss(returns)
    assert torch.isclose(loss, torch.tensor(-0.02), atol=1e-7)

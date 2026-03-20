"""Loss functions for diagnostic portfolio optimization."""

from __future__ import annotations

import warnings

import torch


def return_loss(portfolio_returns: torch.Tensor) -> torch.Tensor:
    """Negative mean portfolio return."""

    if portfolio_returns.numel() == 0:
        raise ValueError("portfolio_returns must contain at least one sample.")
    return -portfolio_returns.mean()


def sharpe_loss(
    portfolio_returns: torch.Tensor,
    eps: float = 1e-6,
    min_batch_size: int = 2,
) -> torch.Tensor:
    """Negative Sharpe-style objective with a small-batch fallback."""

    flat_returns = portfolio_returns.reshape(-1)
    if flat_returns.numel() == 0:
        raise ValueError("portfolio_returns must contain at least one sample.")

    mean_return = flat_returns.mean()
    if flat_returns.numel() < min_batch_size:
        warnings.warn(
            "Sharpe loss received too few samples; falling back to return loss.",
            RuntimeWarning,
            stacklevel=2,
        )
        return -mean_return

    volatility = flat_returns.std(unbiased=False)
    if torch.isnan(volatility) or volatility.item() < eps:
        warnings.warn(
            "Sharpe loss encountered near-zero volatility; falling back to return loss.",
            RuntimeWarning,
            stacklevel=2,
        )
        return -mean_return

    return -(mean_return / (volatility + eps))


def build_loss(name: str, portfolio_returns: torch.Tensor) -> torch.Tensor:
    """Dispatch helper used by train.py."""

    normalized = name.lower()
    if normalized == "return":
        return return_loss(portfolio_returns)
    if normalized == "sharpe":
        return sharpe_loss(portfolio_returns)
    raise ValueError(f"Unsupported loss: {name}")

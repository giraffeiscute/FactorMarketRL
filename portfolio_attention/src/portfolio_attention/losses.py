"""Path-based loss functions for portfolio optimization."""

from __future__ import annotations

import warnings
from typing import Literal

import torch


def return_loss(
    portfolio_returns: torch.Tensor,
    mode: Literal["multiplicative", "additive"] = "multiplicative",
) -> torch.Tensor:
    """Negative terminal return over the path.

    Input shape: [T] or [B, T]
    """
    if portfolio_returns.numel() == 0:
        raise ValueError("portfolio_returns must not be empty.")

    # Ensure 2D [B, T]. If 1D [T], treat as [1, T]
    if portfolio_returns.ndim == 1:
        portfolio_returns = portfolio_returns.unsqueeze(0)

    if mode == "multiplicative":
        # terminal = prod(1 + r) - 1
        terminal_returns = torch.prod(1 + portfolio_returns, dim=1) - 1
    else:
        # terminal = sum(r)
        terminal_returns = torch.sum(portfolio_returns, dim=1)

    return -terminal_returns.mean()


def sharpe_loss(
    portfolio_returns: torch.Tensor,
    eps: float = 1e-6,
    min_time_steps: int = 2,
    risk_free_rate: float = 0.0,
    fallback_mode: Literal["multiplicative", "additive"] = "multiplicative",
) -> torch.Tensor:
    """Negative Sharpe Ratio over the path.

    Input shape: [T] or [B, T]
    """
    if portfolio_returns.ndim == 1:
        portfolio_returns = portfolio_returns.unsqueeze(0)

    batch_size, time_steps = portfolio_returns.shape

    if time_steps < min_time_steps:
        warnings.warn(
            f"Sharpe loss received too few time steps ({time_steps}); falling back to return loss.",
            RuntimeWarning,
            stacklevel=2,
        )
        return return_loss(portfolio_returns, mode=fallback_mode)

    excess_returns = portfolio_returns - risk_free_rate
    mean_ret = excess_returns.mean(dim=1)
    std_ret = excess_returns.std(dim=1, unbiased=True)

    # Sharpe = mean / std
    sharpe = mean_ret / (std_ret + eps)

    # Fallback if std is near zero
    valid_mask = std_ret > eps
    if not valid_mask.any():
        return return_loss(portfolio_returns, mode=fallback_mode)

    return -sharpe[valid_mask].mean()


def differential_sharpe_loss(
    portfolio_returns: torch.Tensor,
    eta: float = 0.2,
    A0: float = 0.0,
    B0: float = 1e-4,
    eps: float = 1e-8,
    reduction: Literal["mean", "sum", "last"] = "mean",
) -> torch.Tensor:
    """Negative Differential Sharpe Ratio (DSR) over the path."""
    if portfolio_returns.ndim == 1:
        portfolio_returns = portfolio_returns.unsqueeze(0)

    batch_size, T = portfolio_returns.shape
    device = portfolio_returns.device

    A = torch.full((batch_size,), A0, device=device)
    B = torch.full((batch_size,), B0, device=device)
    scores = []

    for t in range(T):
        Rt = portfolio_returns[:, t]
        delta_A = Rt - A
        delta_B = Rt**2 - B

        # D_t formula
        numerator = B * delta_A - 0.5 * A * delta_B
        denominator = (B - A**2 + eps) ** 1.5
        Dt = numerator / (denominator + eps)
        scores.append(Dt)

        # Update running estimates
        A = A + eta * delta_A
        B = B + eta * delta_B

    all_scores = torch.stack(scores, dim=1)  # [B, T]

    if reduction == "last":
        score = all_scores[:, -1]
    elif reduction == "sum":
        score = all_scores.sum(dim=1)
    else:
        score = all_scores.mean(dim=1)

    return -score.mean()


def sortino_loss(
    portfolio_returns: torch.Tensor,
    target_return: float = 0.0,
    eps: float = 1e-6,
    min_time_steps: int = 2,
    fallback_mode: Literal["multiplicative", "additive"] = "multiplicative",
) -> torch.Tensor:
    """Negative Sortino Ratio over the path."""
    if portfolio_returns.ndim == 1:
        portfolio_returns = portfolio_returns.unsqueeze(0)

    batch_size, time_steps = portfolio_returns.shape
    if time_steps < min_time_steps:
        return return_loss(portfolio_returns, mode=fallback_mode)

    excess = portfolio_returns - target_return
    mean_ret = excess.mean(dim=1)

    downside = torch.min(excess, torch.zeros_like(excess))
    downside_deviation = torch.sqrt((downside**2).mean(dim=1) + eps)

    sortino = mean_ret / (downside_deviation + eps)
    return -sortino.mean()


def max_drawdown_loss(
    portfolio_returns: torch.Tensor,
    mode: Literal["multiplicative", "additive"] = "multiplicative",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Maximum Drawdown Risk (Positive value, no negative sign)."""
    if portfolio_returns.ndim == 1:
        portfolio_returns = portfolio_returns.unsqueeze(0)

    if mode == "multiplicative":
        equity = torch.cumprod(1 + portfolio_returns, dim=1)
    else:
        equity = 1 + torch.cumsum(portfolio_returns, dim=1)

    running_peak = torch.cummax(equity, dim=1)[0]
    drawdown = (running_peak - equity) / (running_peak + eps)
    max_dd = drawdown.max(dim=1)[0]

    return max_dd.mean()


def cvar_loss(
    portfolio_returns: torch.Tensor,
    alpha: float = 0.05,
) -> torch.Tensor:
    """Conditional Value at Risk (Expected Shortfall) Risk."""
    if portfolio_returns.ndim == 1:
        portfolio_returns = portfolio_returns.unsqueeze(0)

    losses = -portfolio_returns
    batch_size, T = losses.shape

    cvar_list = []
    for i in range(batch_size):
        path_losses = losses[i]
        var = torch.quantile(path_losses, 1 - alpha)
        tail_losses = path_losses[path_losses >= var]
        if tail_losses.numel() == 0:
            cvar_list.append(path_losses.max())
        else:
            cvar_list.append(tail_losses.mean())

    return torch.stack(cvar_list).mean()


def build_loss(
    name: str,
    portfolio_returns: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Dispatch helper for path-based losses."""
    normalized = name.lower().replace("_", "")

    # Terminal Return aliases
    if normalized in {"return", "totalreturn", "terminalreturn"}:
        return return_loss(portfolio_returns, **kwargs)

    # Sharpe aliases
    if normalized in {"sharpe", "sr"}:
        return sharpe_loss(portfolio_returns, **kwargs)

    # DSR aliases
    if normalized in {"dsr", "differentialsharpe"}:
        return differential_sharpe_loss(portfolio_returns, **kwargs)

    # Sortino
    if normalized == "sortino":
        return sortino_loss(portfolio_returns, **kwargs)

    # MDD aliases
    if normalized in {"mdd", "maxdrawdown"}:
        return max_drawdown_loss(portfolio_returns, **kwargs)

    # CVaR
    if normalized == "cvar":
        return cvar_loss(portfolio_returns, **kwargs)

    raise ValueError(f"Unsupported loss: {name}")

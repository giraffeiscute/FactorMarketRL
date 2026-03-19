import torch
import numpy as np


class PerformanceMetrics:
    def __init__(self, trading_days=252, risk_free_rate=0.0):
        self.trading_days = trading_days
        self.risk_free_rate = risk_free_rate / trading_days

    def calculate_metrics(self, portfolio_returns, daily_stock_returns):
        # --- 若輸入是 torch.Tensor，先轉成 numpy ---
        if isinstance(portfolio_returns, torch.Tensor):
            portfolio_returns = portfolio_returns.detach().cpu().numpy()

        if isinstance(daily_stock_returns, torch.Tensor):
            daily_stock_returns = daily_stock_returns.detach().cpu().numpy()

        # --- 若投資組合報酬不是一維，攤平成一維 ---
        if portfolio_returns.ndim > 1:
            portfolio_returns = portfolio_returns.flatten()

        # --- 若波動幾乎為 0，直接回傳 0，避免除以極小值 ---
        if np.std(portfolio_returns) < 1e-9:
            return {
                'Annualized Return': 0,
                'Sharpe Ratio': 0,
                'Sortino Ratio': 0,
                'Max Drawdown': 0,
                'CVaR (5%)': 0,
                'Beta': 0,
                'IR': 0
            }

        # --- 年化報酬 ---
        mean_portfolio_return = np.mean(portfolio_returns)
        annualized_return = mean_portfolio_return * self.trading_days

        # --- 夏普比率 ---
        portfolio_std = np.std(portfolio_returns)
        sharpe_ratio = (
            (mean_portfolio_return - self.risk_free_rate)
            / portfolio_std
            * np.sqrt(self.trading_days)
        )

        # --- 索提諾比率 ---
        negative_returns = portfolio_returns[portfolio_returns < self.risk_free_rate]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0

        if downside_std > 0:
            sortino_ratio = (
                (mean_portfolio_return - self.risk_free_rate)
                / downside_std
                * np.sqrt(self.trading_days)
            )
        else:
            sortino_ratio = 0

        # --- 最大回撤 ---
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        # --- CVaR (5%) ---
        var_5 = np.percentile(portfolio_returns, 5)
        cvar_5 = np.mean(portfolio_returns[portfolio_returns <= var_5])

        # --- Beta ---
        market_returns = np.mean(daily_stock_returns, axis=0)
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        beta = covariance / market_variance if market_variance > 0 else 0

        # --- 回傳結果 ---
        return {
            'Annualized Return': annualized_return,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'CVaR (5%)': -cvar_5,
            'Beta': beta,
            'IR': sharpe_ratio
        }
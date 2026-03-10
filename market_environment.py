import torch
import numpy as np


class MarketEnvironment:
    """
    模拟包含多种波动率情景的股票市场环境
    """
    def __init__(self, trading_days=252):
        self.num_stocks_per_regime = 2001
        self.num_stocks = self.num_stocks_per_regime * 3
        self.d_model = 64
        self.trading_days = trading_days

        # 生成固定的股票嵌入向量
        self.stock_embeddings = torch.randn(self.num_stocks, self.d_model)

        # 设定股票的年化收益和波动率
        self.annual_returns = self._generate_returns()
        self.annual_volatility = self._generate_volatilities()

        # 转换为日度数据
        self.daily_returns_mean = self.annual_returns / self.trading_days
        self.daily_volatility = self.annual_volatility / np.sqrt(self.trading_days)

    def _generate_returns(self):
        """
        为所有三个情景生成年化收益率
        """
        returns = np.zeros(self.num_stocks)
        for i in range(3):
            start_idx = i * self.num_stocks_per_regime
            end_idx = (i + 1) * self.num_stocks_per_regime
            returns[start_idx:end_idx] = self._generate_single_regime_returns()
        return returns

    def _generate_single_regime_returns(self):
        """
        根据规则为单个情景（2001只股票）生成年化收益率
        中心点: 15%, 边缘: -5%
        """
        returns = np.zeros(self.num_stocks_per_regime)
        center_stock = self.num_stocks_per_regime // 2
        max_return = 0.15
        min_return = -0.05

        for i in range(self.num_stocks_per_regime):
            distance_from_center = abs(i - center_stock)
            returns[i] = max_return - distance_from_center * (max_return - min_return) / center_stock
        return returns

    def _generate_volatilities(self):
        """
        为三个情景生成固定的年化波动率
        情景1: 5%, 情景2: 10%, 情景3: 15%
        """
        volatilities = np.zeros(self.num_stocks)
        vol_levels = [0.05, 0.10, 0.15]  # 5%, 10%, 15%
        for i in range(3):
            start_idx = i * self.num_stocks_per_regime
            end_idx = (i + 1) * self.num_stocks_per_regime
            volatilities[start_idx:end_idx] = vol_levels[i]
        return volatilities

    def get_daily_returns(self, num_days):
        """
        生成指定天数的随机日度收益
        """
        daily_returns = np.random.normal(
            loc=self.daily_returns_mean[:, np.newaxis],
            scale=self.daily_volatility[:, np.newaxis],
            size=(self.num_stocks, num_days)
        )
        return torch.tensor(daily_returns, dtype=torch.float32)
"""
這個模組負責生成每支股票固定不變的 `alpha_i`。

它的用途是：
- 依照常態分配為每支股票抽出一個固定效果
- 讓報酬方程式中除了 beta * factor 與 noise 之外，還有個股自己的基礎偏移

主要輸入：
- `stock_ids`：股票代號清單
- `mu_alpha`：alpha 的平均數
- `sigma_alpha`：alpha 的標準差
- `rng`：NumPy 亂數產生器

主要輸出：
- `alpha_df`，欄位為 `[stock_id, alpha]`
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def generate_alpha(
    stock_ids: Sequence[str],
    mu_alpha: float,
    sigma_alpha: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """依常態分配生成股票層級的固定效果 `alpha_i`。"""

    alpha_values = rng.normal(loc=mu_alpha, scale=sigma_alpha, size=len(stock_ids))
    return pd.DataFrame({"stock_id": list(stock_ids), "alpha": alpha_values})

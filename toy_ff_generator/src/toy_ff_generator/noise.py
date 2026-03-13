"""
這個模組負責生成個股噪音 `epsilon_{i,t}`。

支援：
- 所有股票共用同一個 `sigma_epsilon`
- 每支股票手動指定自己的 `sigma_epsilon_i`
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def _coerce_per_stock_sigma(
    stock_ids: Sequence[str],
    per_stock_sigma_epsilon_i: Sequence[float] | Mapping[str, float],
) -> list[float]:
    """把 per-stock noise scale 轉成與 `stock_ids` 對齊的清單。"""

    if isinstance(per_stock_sigma_epsilon_i, Mapping):
        missing_ids = [
            stock_id for stock_id in stock_ids if stock_id not in per_stock_sigma_epsilon_i
        ]
        if missing_ids:
            raise ValueError(
                f"Missing stock ids in per_stock_sigma_epsilon_i: {missing_ids}."
            )
        return [float(per_stock_sigma_epsilon_i[stock_id]) for stock_id in stock_ids]

    if len(per_stock_sigma_epsilon_i) != len(stock_ids):
        raise ValueError(
            "per_stock_sigma_epsilon_i length must equal number of stocks. "
            f"Received length={len(per_stock_sigma_epsilon_i)}, N={len(stock_ids)}."
        )
    return [float(value) for value in per_stock_sigma_epsilon_i]


def _build_sigma_table(
    stock_ids: Sequence[str],
    use_shared_sigma_epsilon: bool,
    shared_sigma_epsilon: float,
    per_stock_sigma_epsilon_i: Sequence[float] | Mapping[str, float] | None,
) -> pd.DataFrame:
    """建立每支股票對應的 `sigma_epsilon_i` 表。"""

    if use_shared_sigma_epsilon:
        return pd.DataFrame(
            {"stock_id": list(stock_ids), "sigma_epsilon_i": float(shared_sigma_epsilon)}
        )

    if per_stock_sigma_epsilon_i is None:
        raise ValueError(
            "per_stock_sigma_epsilon_i is required when use_shared_sigma_epsilon is False."
        )

    sigma_values = _coerce_per_stock_sigma(stock_ids, per_stock_sigma_epsilon_i)
    return pd.DataFrame({"stock_id": list(stock_ids), "sigma_epsilon_i": sigma_values})


def generate_noise(
    stock_ids: Sequence[str],
    time_columns: Sequence[str],
    use_shared_sigma_epsilon: bool,
    shared_sigma_epsilon: float,
    rng: np.random.Generator,
    per_stock_sigma_epsilon_i: Sequence[float] | Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """生成個股噪音 panel，欄位為 `[stock_id, t, epsilon]`。"""

    sigma_df = _build_sigma_table(
        stock_ids=stock_ids,
        use_shared_sigma_epsilon=use_shared_sigma_epsilon,
        shared_sigma_epsilon=shared_sigma_epsilon,
        per_stock_sigma_epsilon_i=per_stock_sigma_epsilon_i,
    )

    rows: list[dict[str, float | str]] = []
    for row in sigma_df.itertuples(index=False):
        draws = rng.normal(loc=0.0, scale=float(row.sigma_epsilon_i), size=len(time_columns))
        for time_label, epsilon in zip(time_columns, draws, strict=True):
            rows.append(
                {
                    "stock_id": row.stock_id,
                    "t": time_label,
                    "epsilon": float(epsilon),
                }
            )

    return pd.DataFrame(rows)

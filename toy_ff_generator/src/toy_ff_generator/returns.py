"""
這個模組負責把所有生成好的元件組裝成最終報酬與價格資料。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from toy_ff_generator.characteristics import FIRM_CHARACTERISTIC_COLUMNS


def build_panel(
    firm_characteristics_df: pd.DataFrame,
    beta_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    epsilon_df: pd.DataFrame,
    factor_df: pd.DataFrame,
) -> pd.DataFrame:
    """把各個生成模組的輸出合併成 long panel，包含每期市場 state。"""

    missing_columns = [
        column_name
        for column_name in FIRM_CHARACTERISTIC_COLUMNS
        if column_name not in firm_characteristics_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "firm_characteristics_df is missing required columns "
            f"{missing_columns}. Expected {FIRM_CHARACTERISTIC_COLUMNS}."
        )

    panel_df = firm_characteristics_df.merge(beta_df, on=["stock_id", "t"], how="inner")
    panel_df = panel_df.merge(alpha_df, on="stock_id", how="inner")
    panel_df = panel_df.merge(epsilon_df, on=["stock_id", "t"], how="inner")
    panel_df = panel_df.merge(factor_df, on="t", how="inner")

    return panel_df[
        [
            "stock_id",
            "t",
            "state",
            *FIRM_CHARACTERISTIC_COLUMNS,
            "alpha",
            "beta_mkt",
            "beta_smb",
            "beta_hml",
            "MKT",
            "SMB",
            "HML",
            "epsilon",
        ]
    ].copy()


def compute_raw_returns(panel_df: pd.DataFrame) -> pd.DataFrame:
    """依照模型方程式計算 clipping 之前的 `raw_return`。"""

    result_df = panel_df.copy()
    result_df["raw_return"] = (
        result_df["alpha"]
        + result_df["beta_mkt"] * result_df["MKT"]
        + result_df["beta_smb"] * result_df["SMB"]
        + result_df["beta_hml"] * result_df["HML"]
        + result_df["epsilon"]
    )
    return result_df


def clip_returns(panel_df: pd.DataFrame, limit_down: float, limit_up: float) -> pd.DataFrame:
    """把 `raw_return` 套用上下限，得到最終觀察到的 `return`。"""

    result_df = panel_df.copy()
    result_df["return"] = result_df["raw_return"].clip(lower=limit_down, upper=limit_up)
    return result_df


def generate_prices(
    panel_df: pd.DataFrame,
    initial_prices: Mapping[str, float],
    time_columns: Sequence[str],
) -> pd.DataFrame:
    """使用 clipped return 依序遞推每支股票的價格路徑。"""

    result_df = panel_df.copy()
    time_order = {time_label: idx for idx, time_label in enumerate(time_columns)}
    result_df["_time_order"] = result_df["t"].map(time_order)
    result_df = result_df.sort_values(["stock_id", "_time_order"]).copy()

    prices: list[float] = []
    for stock_id, stock_panel in result_df.groupby("stock_id", sort=False):
        current_price = float(initial_prices[stock_id])
        for clipped_return in stock_panel["return"].tolist():
            current_price = current_price * (1.0 + float(clipped_return))
            prices.append(current_price)

    result_df["price"] = prices
    return result_df.drop(columns="_time_order")

"""
這個模組負責把所有生成好的元件組裝成最終報酬與價格資料。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from toy_ff_generator.characteristics import FIRM_CHARACTERISTIC_COLUMNS


def _time_label_to_order(time_label: str) -> int:
    """把標準時間標籤 `t_k` 轉成可排序的整數索引。"""

    prefix, _, suffix = time_label.partition("_")
    if prefix != "t" or not suffix.isdigit():
        raise ValueError(
            f"Time label must follow the `t_k` convention. Received {time_label!r}."
        )
    return int(suffix)


def _align_next_period_factor_realizations(factor_df: pd.DataFrame) -> pd.DataFrame:
    """保留當期 state，並把因子 realizations 對齊到下一期。"""

    ordered_factor_df = factor_df.copy()
    ordered_factor_df["_time_order"] = ordered_factor_df["t"].map(_time_label_to_order)
    ordered_factor_df = ordered_factor_df.sort_values("_time_order").reset_index(drop=True)
    ordered_factor_df[["MKT", "SMB", "HML"]] = ordered_factor_df[["MKT", "SMB", "HML"]].shift(-1)

    return ordered_factor_df.iloc[:-1][["t", "state", "MKT", "SMB", "HML"]].copy()


def _align_next_period_epsilon_realizations(epsilon_df: pd.DataFrame) -> pd.DataFrame:
    """把 idiosyncratic noise 對齊到下一期 realizations。"""

    ordered_epsilon_df = epsilon_df.copy()
    ordered_epsilon_df["_time_order"] = ordered_epsilon_df["t"].map(_time_label_to_order)
    ordered_epsilon_df = ordered_epsilon_df.sort_values(["stock_id", "_time_order"]).copy()
    ordered_epsilon_df["epsilon"] = ordered_epsilon_df.groupby("stock_id", sort=False)[
        "epsilon"
    ].shift(-1)

    return ordered_epsilon_df.dropna(subset=["epsilon"])[["stock_id", "t", "epsilon"]].copy()


def build_panel(
    firm_characteristics_df: pd.DataFrame,
    beta_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    epsilon_df: pd.DataFrame,
    factor_df: pd.DataFrame,
) -> pd.DataFrame:
    """把各個生成模組的輸出合併成 long panel，使用 t 的 beta 對應 t+1 的 realizations。"""

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

    next_factor_df = _align_next_period_factor_realizations(factor_df=factor_df)
    next_epsilon_df = _align_next_period_epsilon_realizations(epsilon_df=epsilon_df)

    panel_df = firm_characteristics_df.merge(beta_df, on=["stock_id", "t"], how="inner")
    panel_df = panel_df.merge(alpha_df, on="stock_id", how="inner")
    panel_df = panel_df.merge(next_epsilon_df, on=["stock_id", "t"], how="inner")
    panel_df = panel_df.merge(next_factor_df, on="t", how="inner")

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
    """依照 r_{i,t+1}=alpha_i+beta_{i,t}'X_{t+1}+epsilon_{i,t+1} 計算 `raw_return`。"""

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

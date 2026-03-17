"""Tests for raw return, clipping, and price generation."""

import pandas as pd

from toy_ff_generator.returns import build_panel, clip_returns, compute_raw_returns, generate_prices


def test_returns_pipeline_small_manual_example() -> None:
    panel_df = pd.DataFrame(
        {
            "stock_id": ["stock_000", "stock_000", "stock_001", "stock_001"],
            "t": ["t_0", "t_1", "t_0", "t_1"],
            "firm_size": [1.0, 1.0, 1.0, 1.0],
            "book_to_price": [1.0, 1.0, 1.0, 1.0],
            "alpha": [0.01, 0.01, -0.02, -0.02],
            "beta_mkt": [1.0, 1.0, 0.5, 0.5],
            "beta_smb": [0.5, 0.5, 0.0, 0.0],
            "beta_hml": [-0.5, -0.5, 0.0, 0.0],
            "MKT": [0.02, 0.03, 0.40, -0.30],
            "SMB": [0.01, 0.02, 0.0, 0.0],
            "HML": [0.04, -0.01, 0.0, 0.0],
            "epsilon": [0.005, -0.005, 0.0, 0.0],
        }
    )

    raw_df = compute_raw_returns(panel_df)
    clipped_df = clip_returns(raw_df, limit_down=-0.10, limit_up=0.10)
    price_df = generate_prices(
        clipped_df,
        initial_prices={"stock_000": 100.0, "stock_001": 100.0},
        time_columns=["t_0", "t_1"],
    )

    expected_raw = [0.02, 0.05, 0.18, -0.17]
    expected_clipped = [0.02, 0.05, 0.10, -0.10]
    expected_prices = [102.0, 107.1, 110.0, 99.0]

    assert raw_df["raw_return"].round(10).tolist() == expected_raw
    assert clipped_df["return"].round(10).tolist() == expected_clipped
    assert price_df["price"].round(10).tolist() == expected_prices


def test_build_panel_uses_next_period_factor_and_epsilon_realizations() -> None:
    firm_characteristics_df = pd.DataFrame(
        {
            "stock_id": ["stock_000", "stock_000"],
            "t": ["t_0", "t_1"],
            "firm_size": [1.0, 1.1],
            "book_to_price": [0.8, 0.9],
        }
    )
    beta_df = pd.DataFrame(
        {
            "stock_id": ["stock_000", "stock_000"],
            "t": ["t_0", "t_1"],
            "beta_mkt": [1.0, 2.0],
            "beta_smb": [0.0, 0.0],
            "beta_hml": [0.0, 0.0],
        }
    )
    alpha_df = pd.DataFrame({"stock_id": ["stock_000"], "alpha": [0.0]})
    epsilon_df = pd.DataFrame(
        {
            "stock_id": ["stock_000", "stock_000", "stock_000"],
            "t": ["t_0", "t_1", "t_2"],
            "epsilon": [0.1, 0.2, 0.3],
        }
    )
    factor_df = pd.DataFrame(
        {
            "t": ["t_0", "t_1", "t_2"],
            "state": [-1, 0, 1],
            "MKT": [1.0, 10.0, 100.0],
            "SMB": [2.0, 20.0, 200.0],
            "HML": [3.0, 30.0, 300.0],
        }
    )

    panel_df = build_panel(
        firm_characteristics_df=firm_characteristics_df,
        beta_df=beta_df,
        alpha_df=alpha_df,
        epsilon_df=epsilon_df,
        factor_df=factor_df,
    ).sort_values("t")

    assert panel_df["state"].tolist() == [-1, 0]
    assert panel_df["MKT"].tolist() == [10.0, 100.0]
    assert panel_df["SMB"].tolist() == [20.0, 200.0]
    assert panel_df["HML"].tolist() == [30.0, 300.0]
    assert panel_df["epsilon"].tolist() == [0.2, 0.3]

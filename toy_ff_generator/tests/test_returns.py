"""Tests for raw return, clipping, and price generation."""

import pandas as pd

from toy_ff_generator.returns import clip_returns, compute_raw_returns, generate_prices


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

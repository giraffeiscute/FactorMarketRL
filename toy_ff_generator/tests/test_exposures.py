"""Tests for exposure generation."""

import pandas as pd

from toy_ff_generator.exposures import generate_exposures


def test_generate_exposures_follow_shared_linear_rule() -> None:
    characteristic_df = pd.DataFrame(
        {
            "stock_id": ["stock_000", "stock_001", "stock_002"],
            "t": ["t_0", "t_0", "t_0"],
            "C": [1.0, -2.0, 0.5],
        }
    )

    beta_df = generate_exposures(
        characteristic_df=characteristic_df,
        a_mkt=0.4,
        b_mkt=1.1,
        a_smb=-0.2,
        b_smb=0.3,
        a_hml=0.5,
        b_hml=-0.1,
    )

    assert all(beta_df["beta_mkt"] == 0.4 * characteristic_df["C"] + 1.1)
    assert all(beta_df["beta_smb"] == -0.2 * characteristic_df["C"] + 0.3)
    assert all(beta_df["beta_hml"] == 0.5 * characteristic_df["C"] - 0.1)
    assert all((beta_df["beta_mkt"] - 0.4 * characteristic_df["C"]).round(10) == 1.1)
    assert all((beta_df["beta_smb"] + 0.2 * characteristic_df["C"]).round(10) == 0.3)
    assert all((beta_df["beta_hml"] - 0.5 * characteristic_df["C"]).round(10) == -0.1)

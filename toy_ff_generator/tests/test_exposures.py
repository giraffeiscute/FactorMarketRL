"""Tests for exposure generation."""

import pandas as pd

from toy_ff_generator.exposures import generate_exposures


def test_generate_exposures_follow_vector_linear_rule() -> None:
    characteristic_df = pd.DataFrame(
        {
            "stock_id": ["stock_000", "stock_001", "stock_002"],
            "t": ["t_0", "t_0", "t_0"],
            "C1": [1.0, -2.0, 0.5],
            "C2": [0.0, 1.0, -1.5],
            "C3": [2.0, 0.5, 1.0],
        }
    )

    beta_df = generate_exposures(
        characteristic_df=characteristic_df,
        a_mkt=[0.4, -0.2, 0.1],
        b_mkt=1.1,
        a_smb=[-0.2, 0.3, 0.0],
        b_smb=0.3,
        a_hml=[0.5, 0.1, -0.4],
        b_hml=-0.1,
    )

    expected_beta_mkt = [1.7, 0.15, 1.7]
    expected_beta_smb = [0.1, 1.0, -0.25]
    expected_beta_hml = [-0.4, -1.2, -0.4]

    assert beta_df["beta_mkt"].round(10).tolist() == expected_beta_mkt
    assert beta_df["beta_smb"].round(10).tolist() == expected_beta_smb
    assert beta_df["beta_hml"].round(10).tolist() == expected_beta_hml

"""Tests for exposure generation from latent firm states."""

import pandas as pd

from toy_ff_generator.exposures import generate_exposures


def test_generate_exposures_follow_latent_state_rule() -> None:
    latent_state_df = pd.DataFrame(
        {
            "stock_id": ["stock_000", "stock_001", "stock_002"],
            "t": ["t_0", "t_0", "t_0"],
            "latent_size_state": [0.0, 2.0, -0.5],
            "latent_book_to_price_state": [-0.2, 0.4, 0.7],
        }
    )

    beta_df = generate_exposures(
        latent_state_df=latent_state_df,
        a_mkt=[0.0, 0.0],
        b_mkt=1.0,
        a_smb=[-1.0, 0.0],
        b_smb=0.0,
        a_hml=[0.0, 1.0],
        b_hml=0.0,
    )

    assert beta_df["beta_mkt"].round(10).tolist() == [1.0, 1.0, 1.0]
    assert beta_df["beta_smb"].round(10).tolist() == [0.0, -2.0, 0.5]
    assert beta_df["beta_hml"].round(10).tolist() == [-0.2, 0.4, 0.7]

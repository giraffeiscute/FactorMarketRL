"""Tests for exposure generation from latent beta-axis states."""

import pandas as pd

from toy_ff_generator.exposures import generate_exposures


def test_generate_exposures_identity_mapping_matches_latent_state_axes() -> None:
    latent_state_df = pd.DataFrame(
        {
            "stock_id": ["stock_000", "stock_001", "stock_002"],
            "t": ["t_0", "t_0", "t_0"],
            "latent_beta_mkt_state": [0.0, 2.0, -0.5],
            "latent_beta_smb_state": [-0.2, 0.4, 0.7],
            "latent_beta_hml_state": [1.2, -1.1, 0.3],
        }
    )

    beta_df = generate_exposures(
        latent_state_df=latent_state_df,
        A=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        b=[0.0, 0.0, 0.0],
    )

    assert beta_df["beta_mkt"].round(10).tolist() == [0.0, 2.0, -0.5]
    assert beta_df["beta_smb"].round(10).tolist() == [-0.2, 0.4, 0.7]
    assert beta_df["beta_hml"].round(10).tolist() == [1.2, -1.1, 0.3]

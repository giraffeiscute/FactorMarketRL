"""Tests for validation around latent states and observable firm characteristics."""

import pandas as pd
import pytest

from toy_ff_generator.validation import (
    validate_exposure_setup,
    validate_firm_characteristics_df,
    validate_latent_characteristic_setup,
    validate_latent_state_df,
)


def test_validate_latent_characteristic_setup_requires_two_dimensional_shared_vectors() -> None:
    with pytest.raises(
        ValueError,
        match=r"Omega must have shape \(2,\) for the latent state order",
    ):
        validate_latent_characteristic_setup(
            N=3,
            latent_characteristic_setup={
                "use_shared_latent_state_params": True,
                "shared_params": {
                    "Omega": [0.6, 0.4, 0.2],
                    "mu_X": [0.0, 0.1],
                    "lambda_X": [0.2, 0.1],
                    "sigma_X": [0.3, 0.2],
                    "X0": [0.0, 0.0],
                },
            },
        )


def test_validate_exposure_setup_requires_two_dimensional_loading_vectors() -> None:
    with pytest.raises(
        ValueError,
        match=r"a_hml must have shape \(2,\) for the latent state order",
    ):
        validate_exposure_setup(
            {
                "a_mkt": [0.0, 0.0],
                "a_smb": [-1.0, 0.0],
                "a_hml": [0.0, 1.0, 0.0],
                "b_mkt": 1.0,
                "b_smb": 0.0,
                "b_hml": 0.0,
            }
        )


def test_validate_latent_state_df_requires_latent_columns() -> None:
    with pytest.raises(
        ValueError,
        match=r"latent_state_df is missing required latent state columns",
    ):
        validate_latent_state_df(
            latent_state_df=pd.DataFrame(
                {
                    "stock_id": ["stock_000"],
                    "t": ["t_0"],
                    "firm_size": [1.0],
                    "book_to_price": [0.0],
                }
            ),
            expected_rows=1,
        )


def test_validate_firm_characteristics_df_requires_positive_observable_values() -> None:
    with pytest.raises(
        ValueError,
        match=r"Observable firm characteristics must be strictly positive",
    ):
        validate_firm_characteristics_df(
            firm_characteristics_df=pd.DataFrame(
                {
                    "stock_id": ["stock_000"],
                    "t": ["t_0"],
                    "firm_size": [1.0],
                    "book_to_price": [0.0],
                }
            ),
            expected_rows=1,
        )

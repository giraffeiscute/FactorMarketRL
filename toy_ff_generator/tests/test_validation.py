"""Tests for validation around latent states and observable characteristics."""

import pandas as pd
import pytest

from toy_ff_generator.validation import (
    validate_alpha_epsilon_mode_setup,
    validate_beta_df,
    validate_exposure_setup,
    validate_firm_characteristics_df,
    validate_latent_characteristic_setup,
    validate_latent_state_df,
    validate_mu_class_setup,
)


def test_validate_latent_characteristic_setup_requires_three_dimensional_shared_vectors() -> None:
    with pytest.raises(
        ValueError,
        match=r"Omega must have shape \(3,\) for the latent state order",
    ):
        validate_latent_characteristic_setup(
            N=3,
            latent_characteristic_setup={
                "use_shared_latent_state_params": True,
                "shared_params": {
                    "Omega": [0.6, 0.4],
                    "mu_Z": [0.0, 0.1, 0.2],
                    "lambda_Z": [0.2, 0.1, 0.0],
                    "sigma_Z": [0.3, 0.2, 0.1],
                    "Z0": [0.0, 0.0, 0.0],
                },
            },
        )


def test_validate_exposure_setup_requires_three_dimensional_matrix_and_vector() -> None:
    with pytest.raises(
        ValueError,
        match=r"A must have shape \(3, 3\)",
    ):
        validate_exposure_setup(
            {
                "A": [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                "b": [0.0, 0.0, 0.0],
            }
        )


def test_validate_mu_class_setup_requires_low_mid_high_centers() -> None:
    with pytest.raises(
        ValueError,
        match=r"class_centers must define",
    ):
        validate_mu_class_setup(
            {
                "class_centers": {
                    "low": -0.5,
                    "mid": 0.0,
                }
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
                    "characteristic_1": [1.0],
                    "characteristic_2": [0.0],
                    "characteristic_3": [0.1],
                }
            ),
            expected_rows=1,
        )


def test_validate_firm_characteristics_df_accepts_three_axis_values() -> None:
    validate_firm_characteristics_df(
        firm_characteristics_df=pd.DataFrame(
            {
                "stock_id": ["stock_000"],
                "t": ["t_0"],
                "characteristic_1": [1.0],
                "characteristic_2": [0.0],
                "characteristic_3": [-0.5],
            }
        ),
        expected_rows=1,
    )


def test_validate_alpha_epsilon_mode_setup_rejects_unknown_group() -> None:
    with pytest.raises(
        ValueError,
        match=r"alpha_group must be one of",
    ):
        validate_alpha_epsilon_mode_setup(
            N=1,
            alpha_epsilon_mode_setup={
                "alpha_group": "ultra",
                "epsilon_group": "low",
                "alpha_levels": {
                    "low": 0.001,
                    "mid": 0.002,
                    "high": 0.003,
                },
                "epsilon_levels": {
                    "low": 0.01,
                    "mid": 0.02,
                    "high": 0.03,
                },
            }
        )


def test_validate_alpha_epsilon_mode_setup_rejects_invalid_per_stock_group_length() -> None:
    with pytest.raises(
        ValueError,
        match=r"per_stock_alpha_groups must have length 2",
    ):
        validate_alpha_epsilon_mode_setup(
            N=2,
            alpha_epsilon_mode_setup={
                "alpha_group": "mid",
                "epsilon_group": "mid",
                "alpha_levels": {
                    "low": 0.001,
                    "mid": 0.002,
                    "high": 0.003,
                },
                "epsilon_levels": {
                    "low": 0.01,
                    "mid": 0.02,
                    "high": 0.03,
                },
                "per_stock_alpha_groups": ["low"],
            },
        )


def test_validate_alpha_epsilon_mode_setup_rejects_invalid_per_stock_group_value() -> None:
    with pytest.raises(
        ValueError,
        match=r"per_stock_epsilon_groups must only contain",
    ):
        validate_alpha_epsilon_mode_setup(
            N=2,
            alpha_epsilon_mode_setup={
                "alpha_group": "mid",
                "epsilon_group": "mid",
                "alpha_levels": {
                    "low": 0.001,
                    "mid": 0.002,
                    "high": 0.003,
                },
                "epsilon_levels": {
                    "low": 0.01,
                    "mid": 0.02,
                    "high": 0.03,
                },
                "per_stock_epsilon_groups": ["low", "ultra"],
            },
        )


def test_validate_beta_df_rejects_null_beta_values() -> None:
    with pytest.raises(
        ValueError,
        match=r"beta_df must not contain null values",
    ):
        validate_beta_df(
            beta_df=pd.DataFrame(
                {
                    "stock_id": ["stock_000"],
                    "t": ["t_0"],
                    "beta_mkt": [1.0],
                    "beta_smb": [None],
                    "beta_hml": [0.5],
                }
            )
        )

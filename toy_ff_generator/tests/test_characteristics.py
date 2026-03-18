"""Tests for three-dimensional latent characteristic state generation."""

import numpy as np

from toy_ff_generator.characteristics import (
    generate_latent_characteristic_states,
    state_to_firm_characteristics,
)
from toy_ff_generator.utils import make_stock_ids, make_time_columns, set_random_seed


def test_generate_latent_states_and_observable_characteristics_follow_centered_rule() -> None:
    stock_ids = make_stock_ids(2)
    time_columns = make_time_columns(3)

    latent_state_df = generate_latent_characteristic_states(
        stock_ids=stock_ids,
        time_columns=time_columns,
        state_sequence=[0, 1, -1],
        use_shared_latent_state_params=True,
        shared_params={
            "Omega": [0.5, 0.25, 0.0],
            "mu_Z": [1.0, 0.0, -1.0],
            "lambda_Z": [0.1, 0.2, 0.3],
            "sigma_Z": [0.0, 0.0, 0.0],
            "Z0": [2.0, 0.0, 1.0],
        },
        rng=set_random_seed(7),
    )

    firm_characteristics_df = state_to_firm_characteristics(latent_state_df=latent_state_df)

    assert len(latent_state_df) == 6
    assert list(latent_state_df.columns) == [
        "stock_id",
        "t",
        "latent_characteristic_1_state",
        "latent_characteristic_2_state",
        "latent_characteristic_3_state",
    ]
    assert list(firm_characteristics_df.columns) == [
        "stock_id",
        "t",
        "characteristic_1",
        "characteristic_2",
        "characteristic_3",
    ]

    latent_stock_0_rows = latent_state_df.loc[
        latent_state_df["stock_id"] == "stock_000",
        [
            "latent_characteristic_1_state",
            "latent_characteristic_2_state",
            "latent_characteristic_3_state",
        ],
    ].round(10)
    assert latent_stock_0_rows.values.tolist() == [
        [1.5, 0.0, -1.0],
        [1.35, 0.2, -0.7],
        [1.075, -0.15, -1.3],
    ]

    firm_stock_0_rows = firm_characteristics_df.loc[
        firm_characteristics_df["stock_id"] == "stock_000",
        [
            "characteristic_1",
            "characteristic_2",
            "characteristic_3",
        ],
    ].to_numpy(dtype=float)
    assert np.allclose(
        firm_stock_0_rows,
        np.asarray(
            [
                [1.5, 0.0, -1.0],
                [1.35, 0.2, -0.7],
                [1.075, -0.15, -1.3],
            ],
            dtype=float,
        ),
    )


def test_generate_latent_states_use_fixed_per_stock_mu_i_as_additive_term() -> None:
    stock_ids = make_stock_ids(1)
    time_columns = make_time_columns(3)

    latent_state_df = generate_latent_characteristic_states(
        stock_ids=stock_ids,
        time_columns=time_columns,
        state_sequence=[0, 1, -1],
        use_shared_latent_state_params=False,
        per_stock_params={
            "Omega_i": [[0.5, 0.25, 0.0]],
            "mu_i": [[1.0, 0.0, -1.0]],
            "lambda_i": [[0.1, 0.2, 0.3]],
            "sigma_Z_i": [[0.0, 0.0, 0.0]],
            "Z0_i": [[2.0, 0.0, 1.0]],
        },
        rng=set_random_seed(7),
    )

    latent_stock_0_rows = latent_state_df.loc[
        latent_state_df["stock_id"] == "stock_000",
        [
            "latent_characteristic_1_state",
            "latent_characteristic_2_state",
            "latent_characteristic_3_state",
        ],
    ].round(10)
    assert latent_stock_0_rows.values.tolist() == [
        [2.0, 0.0, -1.0],
        [2.1, 0.2, -0.7],
        [1.95, -0.15, -1.3],
    ]

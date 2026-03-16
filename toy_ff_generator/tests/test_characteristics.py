"""Tests for latent characteristic state generation and positive mapping."""

import numpy as np

from toy_ff_generator.characteristics import (
    generate_latent_characteristic_states,
    state_to_firm_characteristics,
)
from toy_ff_generator.utils import make_stock_ids, make_time_columns, set_random_seed


def test_generate_latent_states_and_positive_firm_characteristics() -> None:
    stock_ids = make_stock_ids(2)
    time_columns = make_time_columns(3)

    latent_state_df = generate_latent_characteristic_states(
        stock_ids=stock_ids,
        time_columns=time_columns,
        state_sequence=[0, 1, -1],
        use_shared_latent_state_params=True,
        shared_params={
            "Omega": [0.5, 0.5],
            "mu_X": [1.0, 0.0],
            "lambda_X": [0.1, 0.2],
            "sigma_X": [0.0, 0.0],
            "X0": [2.0, 0.0],
        },
        rng=set_random_seed(7),
    )

    firm_characteristics_df = state_to_firm_characteristics(latent_state_df=latent_state_df)

    assert len(latent_state_df) == 6
    assert list(latent_state_df.columns) == [
        "stock_id",
        "t",
        "latent_size_state",
        "latent_book_to_price_state",
    ]
    assert list(firm_characteristics_df.columns) == [
        "stock_id",
        "t",
        "firm_size",
        "book_to_price",
    ]

    latent_stock_0_rows = latent_state_df.loc[
        latent_state_df["stock_id"] == "stock_000",
        ["latent_size_state", "latent_book_to_price_state"],
    ].round(10)
    assert latent_stock_0_rows.values.tolist() == [
        [2.0, 0.0],
        [2.1, 0.2],
        [1.95, -0.1],
    ]

    firm_stock_0_rows = firm_characteristics_df.loc[
        firm_characteristics_df["stock_id"] == "stock_000",
        ["firm_size", "book_to_price"],
    ].to_numpy(dtype=float)
    assert np.all(firm_stock_0_rows > 0.0)
    assert np.allclose(
        firm_stock_0_rows,
        np.exp(np.asarray([[2.0, 0.0], [2.1, 0.2], [1.95, -0.1]], dtype=float)),
    )

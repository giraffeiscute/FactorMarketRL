"""Tests for characteristic generation."""

from toy_ff_generator.characteristics import generate_characteristics
from toy_ff_generator.utils import make_stock_ids, make_time_columns, set_random_seed


def test_generate_characteristics_shape_columns_and_inertia() -> None:
    stock_ids = make_stock_ids(2)
    time_columns = make_time_columns(3)

    characteristic_df = generate_characteristics(
        stock_ids=stock_ids,
        time_columns=time_columns,
        state_sequence=[0, 1, -1],
        use_shared_characteristic_params=True,
        shared_params={
            "Omega": 0.5,
            "mu_C": 1.0,
            "Lambda_C": 0.2,
            "sigma_C": 0.0,
            "C0": 2.0,
        },
        rng=set_random_seed(7),
    )

    assert len(characteristic_df) == 6
    assert list(characteristic_df.columns) == ["stock_id", "t", "C"]
    assert set(characteristic_df["stock_id"]) == set(stock_ids)
    assert set(characteristic_df["t"]) == set(time_columns)

    stock_0_values = characteristic_df.loc[
        characteristic_df["stock_id"] == "stock_000", "C"
    ].round(10).tolist()
    assert stock_0_values == [2.0, 2.2, 1.9]

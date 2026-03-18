import numpy as np

from toy_ff_generator.alpha import generate_alpha
from toy_ff_generator.noise import generate_noise
from toy_ff_generator.utils import make_stock_ids, make_time_columns, set_random_seed


def test_generate_alpha_uses_configured_group_value_for_all_stocks() -> None:
    alpha_df = generate_alpha(
        stock_ids=make_stock_ids(4),
        alpha_group="mid",
        alpha_levels={
            "low": 0.001,
            "mid": 0.002,
            "high": 0.003,
        },
    )

    assert alpha_df["alpha"].tolist() == [0.002, 0.002, 0.002, 0.002]


def test_generate_alpha_uses_per_stock_group_values() -> None:
    alpha_df = generate_alpha(
        stock_ids=make_stock_ids(3),
        alpha_group="mid",
        alpha_levels={
            "low": 0.001,
            "mid": 0.002,
            "high": 0.003,
        },
        per_stock_alpha_groups=["low", "mid", "high"],
    )

    assert alpha_df["alpha"].tolist() == [0.001, 0.002, 0.003]


def test_generate_noise_redraws_each_period_with_selected_sigma() -> None:
    epsilon_df = generate_noise(
        stock_ids=make_stock_ids(2),
        time_columns=make_time_columns(3),
        epsilon_group="low",
        epsilon_levels={
            "low": 0.01,
            "mid": 0.02,
            "high": 0.03,
        },
        rng=set_random_seed(7),
    )

    assert len(epsilon_df) == 6
    assert epsilon_df["stock_id"].tolist() == [
        "stock_000",
        "stock_000",
        "stock_000",
        "stock_001",
        "stock_001",
        "stock_001",
    ]
    assert np.allclose(
        epsilon_df["epsilon"].to_numpy(dtype=float),
        [
            0.00001230153357482574,
            0.002987455375084698,
            -0.002741378553622176,
            -0.00890591838778926,
            -0.004546707851717225,
            -0.009916465549964623,
        ],
    )


def test_generate_noise_uses_per_stock_sigma_groups() -> None:
    epsilon_df = generate_noise(
        stock_ids=make_stock_ids(2),
        time_columns=make_time_columns(3),
        epsilon_group="mid",
        epsilon_levels={
            "low": 0.01,
            "mid": 0.02,
            "high": 0.03,
        },
        rng=set_random_seed(7),
        per_stock_epsilon_groups=["low", "high"],
    )

    assert np.allclose(
        epsilon_df["epsilon"].to_numpy(dtype=float),
        [
            0.00001230153357482574,
            0.002987455375084698,
            -0.002741378553622176,
            -0.02671775516336778,
            -0.013640123555151674,
            -0.02974939664989387,
        ],
    )

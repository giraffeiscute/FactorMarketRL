"""Tests for factor generation."""

from toy_ff_generator.factors import generate_factors
from toy_ff_generator.utils import set_random_seed


def test_generate_factors_columns_length_and_reproducibility() -> None:
    df_first = generate_factors(
        t_count=8,
        state_sequence=[0, 1, 1, 0, -1, -1, 0, 1],
        X0=[0.0, 0.0, 0.0],
        Phi=[
            [0.4, 0.1, 0.0],
            [0.0, 0.3, 0.1],
            [0.0, 0.0, 0.2],
        ],
        Delta=[0.01, 0.00, -0.01],
        Sigma_X_bear=[
            [0.02, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.03],
        ],
        Sigma_X_neutral=[
            [0.02, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.03],
        ],
        Sigma_X_bull=[
            [0.02, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.03],
        ],
        rng=set_random_seed(123),
    )
    df_second = generate_factors(
        t_count=8,
        state_sequence=[0, 1, 1, 0, -1, -1, 0, 1],
        X0=[0.0, 0.0, 0.0],
        Phi=[
            [0.4, 0.1, 0.0],
            [0.0, 0.3, 0.1],
            [0.0, 0.0, 0.2],
        ],
        Delta=[0.01, 0.00, -0.01],
        Sigma_X_bear=[
            [0.02, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.03],
        ],
        Sigma_X_neutral=[
            [0.02, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.03],
        ],
        Sigma_X_bull=[
            [0.02, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.03],
        ],
        rng=set_random_seed(123),
    )

    assert list(df_first.columns) == ["t", "MKT", "SMB", "HML"]
    assert len(df_first) == 8
    assert df_first.equals(df_second)

"""Tests for factor generation."""

import numpy as np

from toy_ff_generator.factors import generate_factors
from toy_ff_generator.utils import set_random_seed


def test_generate_factors_columns_length_and_reproducibility() -> None:
    kwargs = {
        "t_count": 8,
        "state_sequence": [0, 1, 1, 0, -1, -1, 0, 1],
        "X0": [0.0, 0.0, 0.0],
        "Phi": [
            [0.4, 0.1, 0.0],
            [0.0, 0.3, 0.1],
            [0.0, 0.0, 0.2],
        ],
        "mu_bear": [-0.01, -0.02, 0.01],
        "mu_neutral": [0.001, 0.0, 0.0],
        "mu_bull": [0.02, 0.01, -0.01],
        "Delta": None,
        "Sigma_X_bear": [
            [0.02, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.03],
        ],
        "Sigma_X_neutral": [
            [0.02, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.03],
        ],
        "Sigma_X_bull": [
            [0.02, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.03],
        ],
    }

    df_first = generate_factors(rng=set_random_seed(123), **kwargs)
    df_second = generate_factors(rng=set_random_seed(123), **kwargs)

    assert list(df_first.columns) == ["t", "MKT", "SMB", "HML"]
    assert len(df_first) == 8
    assert df_first.equals(df_second)


def test_generate_factors_all_neutral_mean_tracks_stationary_mean() -> None:
    phi = np.asarray(
        [
            [0.45, 0.05, 0.02],
            [0.04, 0.30, 0.03],
            [0.02, 0.04, 0.25],
        ],
        dtype=float,
    )
    mu_neutral = np.asarray([0.0003, 0.0, 0.0], dtype=float)

    factor_df = generate_factors(
        t_count=6000,
        state_sequence=[0] * 6000,
        X0=[0.0, 0.0, 0.0],
        Phi=phi,
        mu_bear=[-0.01, -0.003, 0.001],
        mu_neutral=mu_neutral,
        mu_bull=[0.008, 0.002, -0.001],
        Delta=None,
        Sigma_X_bear=[
            [0.0040, 0.00035, 0.00020],
            [0.00035, 0.00100, 0.00018],
            [0.00020, 0.00018, 0.00080],
        ],
        Sigma_X_neutral=[
            [0.0012, 0.00015, 0.00008],
            [0.00015, 0.00065, 0.00010],
            [0.00008, 0.00010, 0.00055],
        ],
        Sigma_X_bull=[
            [0.0009, 0.00018, 0.00005],
            [0.00018, 0.00055, 0.00006],
            [0.00005, 0.00006, 0.00045],
        ],
        rng=set_random_seed(424),
    )

    expected_stationary_mean = np.linalg.solve(np.eye(3) - phi, mu_neutral)
    sample_mean = factor_df.loc[1000:, ["MKT", "SMB", "HML"]].mean().to_numpy(dtype=float)

    assert sample_mean[0] > 0.0
    assert np.allclose(sample_mean, expected_stationary_mean, atol=0.003)

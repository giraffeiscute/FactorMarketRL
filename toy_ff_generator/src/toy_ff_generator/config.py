from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_ORDER = (-1, 0, 1)
STATE_NAME_MAP = {-1: "bear", 0: "neutral", 1: "bull"}
MU_CLASS_LABELS = ("low", "mid", "high")
MU_AXES = ("characteristic_1", "characteristic_2", "characteristic_3")
PROFILE_GROUP_LABELS = ("mid", "low", "high")


def _default_mu_class_centers() -> dict[str, float]:
    """Return numeric centers for the three fixed mu_i class labels."""

    return {
        "low": -0.5,
        "mid": 0.0,
        "high": 0.5,
    }


def _default_mu_class_triplets(n: int) -> list[tuple[str, str, str]]:
    """Build a deterministic stock universe by cycling over 27 fixed mu_i triplets."""

    all_triplets = list(product(MU_CLASS_LABELS, repeat=len(MU_AXES)))
    return [all_triplets[idx % len(all_triplets)] for idx in range(n)]


def _default_stock_profiles(
    n: int,
) -> list[tuple[tuple[str, str, str], str, str]]:
    """Build deterministic contiguous stock blocks over the 243 base profiles."""

    mu_triplets = _default_mu_class_triplets(len(MU_CLASS_LABELS) ** len(MU_AXES))
    all_profiles = [
        (mu_triplet, alpha_group, epsilon_group)
        for alpha_group in PROFILE_GROUP_LABELS
        for epsilon_group in PROFILE_GROUP_LABELS
        for mu_triplet in mu_triplets
    ]
    base_profile_count = len(all_profiles)
    base_block_size = n // base_profile_count
    remainder = n % base_profile_count

    stock_profiles: list[tuple[tuple[str, str, str], str, str]] = []
    for idx, profile in enumerate(all_profiles):
        block_size = base_block_size + (1 if idx < remainder else 0)
        stock_profiles.extend([profile] * block_size)

    return stock_profiles


def _default_per_stock_alpha_epsilon_groups(n: int) -> dict[str, list[str]]:
    """Build deterministic per-stock alpha/epsilon group assignments."""

    stock_profiles = _default_stock_profiles(n)
    return {
        "per_stock_alpha_groups": [alpha_group for _, alpha_group, _ in stock_profiles],
        "per_stock_epsilon_groups": [epsilon_group for _, _, epsilon_group in stock_profiles],
    }


def _triplets_to_mu_vectors(
    triplets: list[tuple[str, str, str]],
    class_centers: dict[str, float],
) -> list[list[float]]:
    """Map fixed mu_i class triplets to fixed per-stock three-dimensional vectors."""

    return [
        [class_centers[class_label] for class_label in triplet]
        for triplet in triplets
    ]


def _default_per_stock_latent_state_params(n: int) -> dict[str, list[list[float]]]:
    """Build deterministic per-stock latent characteristic-state parameters with fixed mu_i."""

    class_centers = _default_mu_class_centers()
    mu_class_triplets = [mu_triplet for mu_triplet, _, _ in _default_stock_profiles(n)]
    mu_i = _triplets_to_mu_vectors(mu_class_triplets, class_centers)

    return {
        "Omega_i": [[0.65, 0.65, 0.65] for _ in range(n)],
        "mu_i": mu_i,
        "lambda_i": [[0.08, 0.05, 0.05] for _ in range(n)],
        "sigma_Z_i": [[0.06, 0.06, 0.06] for _ in range(n)],
        "Z0_i": [list(mu_vector) for mu_vector in mu_i],
    }


def _default_per_stock_initial_prices(n: int) -> list[float]:
    """Build deterministic initial prices."""

    return [100.0 + 2.5 * idx for idx in range(n)]


def build_default_config() -> dict[str, Any]:
    """Build the project default simulation config."""

    N = 5460
    T = 40
    return {
        "simulation_setup": {
            "N": N,
            "T": T,
            "random_seed": 42,
        },
        "market_state_setup": {
            "state_sequence": None,
            "initial_state": -1,
            "transition_matrix": [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
        },
        "factor_vector_ar_setup": {
            "X0": [0.001, 0.0, 0.0],
            "Phi": [
                [0.6, 0.05, 0.02],
                [0.04, 0.55, 0.03],
                [0.02, 0.04, 0.25],
            ],
            "mu_bear": [-0.01, -0.003, 0.001],
            "mu_neutral": [0.0003, 0.0, 0.0],
            "mu_bull": [0.01, 0.01, -0.01],
            "Sigma_X_bear": [
                [0.00040, 0.000035, 0.000020],
                [0.000035, 0.000400, 0.000018],
                [0.000020, 0.000018, 0.000080],
            ],
            "Sigma_X_neutral": [
                [0.00012, 0.000015, 0.000008],
                [0.000015, 0.000065, 0.000010],
                [0.000008, 0.000010, 0.000055],
            ],
            "Sigma_X_bull": [
                [0.00009, 0.000018, 0.000005],
                [0.000018, 0.000055, 0.000006],
                [0.000005, 0.000006, 0.000045],
            ],
        },
        "mu_class_setup": {
            "class_centers": _default_mu_class_centers(),
        },
        "latent_characteristic_setup": {
            "use_shared_latent_state_params": False,
            "shared_params": {
                "Omega": [0.65, 0.65, 0.65],
                "mu_Z": [0.0, 0.0, 0.0],
                "lambda_Z": [0.05, 0.03, 0.03],
                "sigma_Z": [0.05, 0.05, 0.05],
                "Z0": [0.0, 0.0, 0.0],
            },
            "per_stock_params": _default_per_stock_latent_state_params(N),
        },
        "exposure_setup": {
            "A": [
                [0.1, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
            ],
            "b": [1, 0.4, 0.0],
        },
        "alpha_epsilon_mode_setup": {
            "alpha_group": "mid",
            "epsilon_group": "mid",
            "alpha_levels": {
                "low": -0.0001,
                "mid": 0.0002,
                "high": 0.0003,
            },
            "epsilon_levels": {
                "low": 0.01,
                "mid": 0.025,
                "high": 0.04,
            },
            **_default_per_stock_alpha_epsilon_groups(N),
        },
        "clipping_price_setup": {
            "limit_up": 0.10,
            "limit_down": -0.10,
            "shared_init_price": True,
            "initial_price": 100.0,
            "per_stock_initial_price": _default_per_stock_initial_prices(N),
        },
        "output_setup": {
            "output_dir": str(PROJECT_ROOT / "outputs"),
        },
    }

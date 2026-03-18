from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_ORDER = (-1, 0, 1)
STATE_NAME_MAP = {-1: "bear", 0: "neutral", 1: "bull"}
BETA_CLASS_LABELS = ("low", "mid", "high")
BETA_CLASS_AXES = ("mkt", "smb", "hml")


def _default_beta_class_centers() -> dict[str, float]:
    return {
        "low": -0.5,
        "mid": 0.0,
        "high": 0.5,
    }


def _default_beta_class_triplets(n: int) -> list[tuple[str, str, str]]:
    """Build a deterministic stock universe by cycling over 27 true beta classes."""

    all_triplets = list(product(BETA_CLASS_LABELS, repeat=len(BETA_CLASS_AXES)))
    return [all_triplets[idx % len(all_triplets)] for idx in range(n)]


def _triplets_to_mu_vectors(
    triplets: list[tuple[str, str, str]],
    class_centers: dict[str, float],
) -> list[list[float]]:
    return [
        [class_centers[mkt_group], class_centers[smb_group], class_centers[hml_group]]
        for mkt_group, smb_group, hml_group in triplets
    ]


def _default_per_stock_latent_state_params(n: int) -> dict[str, list[list[float]]]:
    """Build deterministic per-stock latent beta-state parameters."""

    class_centers = _default_beta_class_centers()
    beta_class_triplets = _default_beta_class_triplets(n)
    mu_i = _triplets_to_mu_vectors(beta_class_triplets, class_centers)

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

    N = 27
    T = 10
    return {
        "simulation_setup": {
            "N": N,
            "T": T,
            "random_seed": 42,
        },
        "market_state_setup": {
            "state_sequence": None,
            "initial_state": 0,
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
            "mu_bear": [-0.03, -0.003, 0.001],
            "mu_neutral": [0.0003, 0.0, 0.0],
            "mu_bull": [0.005, 0.01, -0.01],
            "Sigma_X_bear": [
                [0.0040, 0.00035, 0.00020],
                [0.00035, 0.00400, 0.00018],
                [0.00020, 0.00018, 0.00080],
            ],
            "Sigma_X_neutral": [
                [0.0012, 0.00015, 0.00008],
                [0.00015, 0.00065, 0.00010],
                [0.00008, 0.00010, 0.00055],
            ],
            "Sigma_X_bull": [
                [0.0009, 0.00018, 0.00005],
                [0.00018, 0.00055, 0.00006],
                [0.00005, 0.00006, 0.00045],
            ],
        },
        "beta_class_setup": {
            "class_centers": _default_beta_class_centers(),
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
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "b": [0.0, 0.0, 0.0],
        },
        "alpha_epsilon_mode_setup": {
            "alpha_group": "mid",
            "epsilon_group": "mid",
            "alpha_levels": {
                "low": -0.01,
                "mid": 0.002,
                "high": 0.01,
            },
            "epsilon_levels": {
                "low": 0.01,
                "mid": 0.02,
                "high": 0.03,
            },
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

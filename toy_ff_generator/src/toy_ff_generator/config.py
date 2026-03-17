from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_ORDER = (-1, 0, 1)
STATE_NAME_MAP = {-1: "bear", 0: "neutral", 1: "bull"}


def _default_per_stock_latent_state_params(n: int) -> dict[str, list[list[float]]]:
    """建立每檔股票的預設 latent characteristic state 參數。"""

    mu_i = np.column_stack(
        [
            np.linspace(0.08, 0.02, n),
            np.linspace(0.05, 0.01, n),
        ]
    ).tolist()

    return {
        "Omega_i": [[0.55, 0.45] for _ in range(n)],
        "mu_i": mu_i,
        "lambda_i": [[0.08, 0.03] for _ in range(n)],
        "sigma_X_i": [[0.35, 0.30] for _ in range(n)],
        "X0_i": [[0.00, 0.00] for _ in range(n)],
    }


def _default_per_stock_sigma_epsilon(n: int) -> list[float]:
    """建立每檔股票的預設 idiosyncratic epsilon 波動設定。"""

    return [0.020 + 0.002 * idx for idx in range(n)]


def _default_per_stock_initial_prices(n: int) -> list[float]:
    """建立每檔股票的預設初始價格。"""

    return [100.0 + 2.5 * idx for idx in range(n)]


def build_default_config() -> dict[str, Any]:
    """建立專案使用的預設模擬參數設定。"""

    N = 5
    T = 10
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
        "latent_characteristic_setup": {
            "use_shared_latent_state_params": False,
            "shared_params": {
                "Omega": [0.95, 0.9],
                "mu_X": [0, 0],
                "lambda_X": [0.05, 0.02],
                "sigma_X": [0.05, 0.02],
                "X0": [0.00, 0.00],
            },
            "per_stock_params": _default_per_stock_latent_state_params(N),
        },
        "exposure_setup": {
            "a_mkt": [0.03, 0.03],
            "a_smb": [-0.712, 0.00],
            "a_hml": [0.00, 0.834],
            "b_mkt": 1.00,
            "b_smb": 0.03,
            "b_hml": -0.465,
        },
        "alpha_setup": {
            "mu_alpha": 0.00,
            "sigma_alpha": 0.005,
        },
        "epsilon_setup": {
            "use_shared_sigma_epsilon": True,
            "shared_sigma_epsilon": 0.03,
            "per_stock_sigma_epsilon_i": _default_per_stock_sigma_epsilon(N),
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
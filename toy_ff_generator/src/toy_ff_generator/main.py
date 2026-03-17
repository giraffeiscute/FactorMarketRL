"""
這個模組是整個 toy FF generator 的主流程。

本次更新重點：
1. 因子保留目前的 3 維向量 AR(1)
2. characteristic 先以二維 latent state 遞迴，再映射成正值的 firm characteristics
3. beta 直接使用二維 latent state 做線性組合
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from toy_ff_generator.alpha import generate_alpha
from toy_ff_generator.characteristics import (
    FIRM_CHARACTERISTIC_COLUMNS,
    generate_latent_characteristic_states,
    state_to_firm_characteristics,
)
from toy_ff_generator.exposures import generate_exposures
from toy_ff_generator.factors import generate_factors
from toy_ff_generator.noise import generate_noise
from toy_ff_generator.returns import (
    build_panel,
    clip_returns,
    compute_raw_returns,
    generate_prices,
)
from toy_ff_generator.utils import (
    make_stock_ids,
    make_time_columns,
    save_outputs,
    set_random_seed,
)
from toy_ff_generator.validation import (
    validate_component_row_count,
    validate_firm_characteristics_df,
    validate_latent_state_df,
    validate_panel_row_count,
    validate_simulation_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_ORDER = (-1, 0, 1)


def _default_per_stock_latent_state_params(n: int) -> dict[str, list[list[float]]]:
    """建立預設的 per-stock latent characteristic state 參數。"""

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
    """建立預設的 per-stock epsilon 波動尺度。"""

    return [0.020 + 0.002 * idx for idx in range(n)]


def _default_per_stock_initial_prices(n: int) -> list[float]:
    """建立預設的 per-stock 初始價格。"""

    return [100.0 + 2.5 * idx for idx in range(n)]


def build_default_config() -> dict[str, Any]:
    """建立整個專案的預設參數字典。"""

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
                [0.04, 0.30, 0.03],
                [0.02, 0.04, 0.25],
            ],
            "mu_bear": [-0.025, -0.003, 0.001],
            "mu_neutral": [0.0003, 0.0, 0.0],
            "mu_bull": [0.008, 0.002, -0.001],
            "Sigma_X_bear": [
                [0.0040, 0.00035, 0.00020],
                [0.00035, 0.00100, 0.00018],
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
                "mu_X": [0.0001, 0.00001],
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
            "sigma_alpha": 0.015,
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


def _build_state_sequence(
    t_count: int,
    market_state_setup: Mapping[str, Any],
    rng: np.random.Generator,
) -> list[int]:
    """建立本次模擬實際使用的 scalar regime sequence。"""

    manual_sequence = market_state_setup.get("state_sequence")
    if manual_sequence is not None:
        return [int(state) for state in manual_sequence]

    transition_matrix = np.asarray(market_state_setup["transition_matrix"], dtype=float)
    current_state = int(market_state_setup["initial_state"])
    state_sequence = [current_state]

    for _ in range(1, t_count):
        current_index = STATE_ORDER.index(current_state)
        current_state = int(rng.choice(STATE_ORDER, p=transition_matrix[current_index]))
        state_sequence.append(current_state)

    return state_sequence


def _extend_state_sequence_for_next_period(
    state_sequence: list[int],
    market_state_setup: Mapping[str, Any],
    rng: np.random.Generator,
) -> list[int]:
    """在決策期 state path 後補上一個 t+1 state，供 next-period reward 使用。"""

    if market_state_setup.get("state_sequence") is not None:
        return [*state_sequence, int(state_sequence[-1])]

    transition_matrix = np.asarray(market_state_setup["transition_matrix"], dtype=float)
    current_state = int(state_sequence[-1])
    current_index = STATE_ORDER.index(current_state)
    next_state = int(rng.choice(STATE_ORDER, p=transition_matrix[current_index]))
    return [*state_sequence, next_state]


def _build_initial_prices(
    stock_ids: list[str],
    clipping_price_setup: Mapping[str, Any],
) -> dict[str, float]:
    """依照價格設定建立每支股票的初始價格。"""

    if clipping_price_setup["shared_init_price"]:
        initial_price = float(clipping_price_setup["initial_price"])
        return {stock_id: initial_price for stock_id in stock_ids}

    per_stock_initial_price = clipping_price_setup["per_stock_initial_price"]
    return {
        stock_id: float(price)
        for stock_id, price in zip(stock_ids, per_stock_initial_price, strict=True)
    }


def _apply_overrides(
    config: dict[str, Any],
    output_dir: str | None = None,
    seed: int | None = None,
    N: int | None = None,
    T: int | None = None,
    S: int | None = None,
) -> dict[str, Any]:
    """把函式傳入的覆寫值套用到預設設定上。"""

    updated = deepcopy(config)

    if output_dir is not None:
        updated["output_setup"]["output_dir"] = output_dir
    if seed is not None:
        updated["simulation_setup"]["random_seed"] = seed
    if N is not None:
        updated["simulation_setup"]["N"] = N
        updated["latent_characteristic_setup"]["per_stock_params"] = _default_per_stock_latent_state_params(N)
        updated["epsilon_setup"]["per_stock_sigma_epsilon_i"] = _default_per_stock_sigma_epsilon(N)
        updated["clipping_price_setup"]["per_stock_initial_price"] = _default_per_stock_initial_prices(N)
    if T is not None:
        updated["simulation_setup"]["T"] = T
    if S is not None:
        updated["market_state_setup"]["state_sequence"] = [S] * updated["simulation_setup"]["T"]
        updated["market_state_setup"]["initial_state"] = S

    return updated


def run_simulation(
    output_dir: str | None = None,
    seed: int | None = None,
    N: int | None = None,
    T: int | None = None,
    S: int | None = None,
) -> dict[str, Any]:
    """執行完整模擬流程，並把輸出檔寫入磁碟。"""

    config = _apply_overrides(
        config=build_default_config(),
        output_dir=output_dir,
        seed=seed,
        N=N,
        T=T,
        S=S,
    )

    simulation_setup = config["simulation_setup"]
    market_state_setup = config["market_state_setup"]
    factor_vector_ar_setup = config["factor_vector_ar_setup"]
    latent_characteristic_setup = config["latent_characteristic_setup"]
    exposure_setup = config["exposure_setup"]
    alpha_setup = config["alpha_setup"]
    epsilon_setup = config["epsilon_setup"]
    clipping_price_setup = config["clipping_price_setup"]
    output_setup = config["output_setup"]

    stock_ids = make_stock_ids(simulation_setup["N"])
    time_columns = make_time_columns(simulation_setup["T"])
    reward_time_columns = make_time_columns(simulation_setup["T"] + 1)
    rng = set_random_seed(simulation_setup["random_seed"])

    state_sequence = _build_state_sequence(
        t_count=simulation_setup["T"],
        market_state_setup=market_state_setup,
        rng=rng,
    )
    reward_state_sequence = _extend_state_sequence_for_next_period(
        state_sequence=state_sequence,
        market_state_setup=market_state_setup,
        rng=rng,
    )
    config["market_state_setup"]["resolved_state_sequence"] = state_sequence
    config["market_state_setup"]["resolved_reward_state_sequence"] = reward_state_sequence

    validate_simulation_inputs(
        N=simulation_setup["N"],
        T=simulation_setup["T"],
        market_state_setup={**market_state_setup, "state_sequence": state_sequence},
        factor_vector_ar_setup=factor_vector_ar_setup,
        latent_characteristic_setup=latent_characteristic_setup,
        exposure_setup=exposure_setup,
        alpha_setup=alpha_setup,
        epsilon_setup=epsilon_setup,
        clipping_price_setup=clipping_price_setup,
    )

    factor_df = generate_factors(
        t_count=simulation_setup["T"] + 1,
        state_sequence=reward_state_sequence,
        X0=factor_vector_ar_setup["X0"],
        Phi=factor_vector_ar_setup["Phi"],
        Delta=factor_vector_ar_setup.get("Delta"),
        Sigma_X_bear=factor_vector_ar_setup["Sigma_X_bear"],
        Sigma_X_neutral=factor_vector_ar_setup["Sigma_X_neutral"],
        Sigma_X_bull=factor_vector_ar_setup["Sigma_X_bull"],
        rng=rng,
        mu_bear=factor_vector_ar_setup.get("mu_bear"),
        mu_neutral=factor_vector_ar_setup.get("mu_neutral"),
        mu_bull=factor_vector_ar_setup.get("mu_bull"),
    )

    latent_state_df = generate_latent_characteristic_states(
        stock_ids=stock_ids,
        time_columns=time_columns,
        state_sequence=state_sequence,
        use_shared_latent_state_params=latent_characteristic_setup[
            "use_shared_latent_state_params"
        ],
        shared_params=latent_characteristic_setup["shared_params"],
        per_stock_params=latent_characteristic_setup["per_stock_params"],
        rng=rng,
    )
    validate_latent_state_df(
        latent_state_df=latent_state_df,
        expected_rows=simulation_setup["N"] * simulation_setup["T"],
    )

    firm_characteristics_df = state_to_firm_characteristics(latent_state_df=latent_state_df)
    validate_firm_characteristics_df(
        firm_characteristics_df=firm_characteristics_df,
        expected_rows=simulation_setup["N"] * simulation_setup["T"],
    )

    beta_df = generate_exposures(
        latent_state_df=latent_state_df,
        a_mkt=exposure_setup["a_mkt"],
        b_mkt=exposure_setup["b_mkt"],
        a_smb=exposure_setup["a_smb"],
        b_smb=exposure_setup["b_smb"],
        a_hml=exposure_setup["a_hml"],
        b_hml=exposure_setup["b_hml"],
    )
    validate_component_row_count(
        name="beta_df",
        df=beta_df,
        expected_rows=simulation_setup["N"] * simulation_setup["T"],
    )

    alpha_df = generate_alpha(
        stock_ids=stock_ids,
        mu_alpha=alpha_setup["mu_alpha"],
        sigma_alpha=alpha_setup["sigma_alpha"],
        rng=rng,
    )

    epsilon_df = generate_noise(
        stock_ids=stock_ids,
        time_columns=reward_time_columns,
        use_shared_sigma_epsilon=epsilon_setup["use_shared_sigma_epsilon"],
        shared_sigma_epsilon=epsilon_setup["shared_sigma_epsilon"],
        per_stock_sigma_epsilon_i=epsilon_setup["per_stock_sigma_epsilon_i"],
        rng=rng,
    )
    validate_component_row_count(
        name="epsilon_df",
        df=epsilon_df,
        expected_rows=simulation_setup["N"] * (simulation_setup["T"] + 1),
    )

    panel_long_df = build_panel(
        firm_characteristics_df=firm_characteristics_df,
        beta_df=beta_df,
        alpha_df=alpha_df,
        epsilon_df=epsilon_df,
        factor_df=factor_df,
    )
    validate_panel_row_count(
        panel_df=panel_long_df,
        expected_rows=simulation_setup["N"] * simulation_setup["T"],
    )
    panel_long_df = compute_raw_returns(panel_long_df)

    panel_long_df = clip_returns(
        panel_df=panel_long_df,
        limit_down=clipping_price_setup["limit_down"],
        limit_up=clipping_price_setup["limit_up"],
    )

    initial_prices = _build_initial_prices(
        stock_ids=stock_ids,
        clipping_price_setup=clipping_price_setup,
    )
    panel_long_df = generate_prices(
        panel_df=panel_long_df,
        initial_prices=initial_prices,
        time_columns=time_columns,
    )
    panel_long_df = panel_long_df[
        [
            "stock_id",
            "t",
            "state",
            *FIRM_CHARACTERISTIC_COLUMNS,
            "alpha",
            "beta_mkt",
            "beta_smb",
            "beta_hml",
            "MKT",
            "SMB",
            "HML",
            "epsilon",
            "raw_return",
            "return",
            "price",
        ]
    ].copy()

    output_paths = save_outputs(
        panel_long_df=panel_long_df,
        firm_characteristics_df=firm_characteristics_df,
        output_dir=output_setup["output_dir"],
        time_columns=time_columns,
        metadata=config,
    )

    return {
        "config": config,
        "state_sequence": state_sequence,
        "factor_df": factor_df,
        "latent_state_df": latent_state_df,
        "firm_characteristics_df": firm_characteristics_df,
        "beta_df": beta_df,
        "alpha_df": alpha_df,
        "epsilon_df": epsilon_df,
        "panel_long_df": panel_long_df,
        "output_paths": output_paths,
    }


def main() -> pd.DataFrame:
    """用預設參數執行主流程，並回傳最終 long panel。"""

    result = run_simulation()
    return result["panel_long_df"]


if __name__ == "__main__":
    main()

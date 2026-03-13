"""
這個模組負責檢查主模擬流程的輸入參數是否合法。

本次特別加強：
- 向量 AR(1) 的 shape / covariance 檢查
- 動態 characteristic 參數完整性檢查
- state sequence / Markov transition matrix 檢查
- merge 後資料筆數檢查
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

STATE_VALUES = (-1, 0, 1)


def _validate_positive(name: str, value: float) -> None:
    """檢查數值是否嚴格大於 0。"""

    if value <= 0:
        raise ValueError(f"{name} must be > 0. Received {value}.")


def _validate_state_values(state_values: Sequence[int], name: str) -> None:
    """檢查 state 內容是否只包含 -1、0、1。"""

    invalid_values = sorted(set(int(value) for value in state_values) - set(STATE_VALUES))
    if invalid_values:
        raise ValueError(
            f"{name} must only contain -1, 0, 1. Received invalid values: {invalid_values}."
        )


def _coerce_named_vector(
    values: Sequence[float] | Mapping[str, float],
    stock_ids: Sequence[str],
    name: str,
) -> np.ndarray:
    """把 per-stock 參數統一轉成與 `stock_ids` 對齊的向量。"""

    if isinstance(values, Mapping):
        missing_ids = [stock_id for stock_id in stock_ids if stock_id not in values]
        if missing_ids:
            raise ValueError(f"{name} is missing stock ids: {missing_ids}.")
        return np.asarray([values[stock_id] for stock_id in stock_ids], dtype=float)

    if len(values) != len(stock_ids):
        raise ValueError(
            f"{name} length must equal number of stocks. "
            f"Received length={len(values)}, N={len(stock_ids)}."
        )
    return np.asarray(values, dtype=float)


def _validate_covariance_matrix(name: str, matrix: Sequence[Sequence[float]]) -> None:
    """檢查 covariance matrix 的 shape、對稱性與半正定性。"""

    array = np.asarray(matrix, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3). Received {array.shape}.")

    if not np.allclose(array, array.T, atol=1e-10):
        raise ValueError(f"{name} must be symmetric.")

    eigenvalues = np.linalg.eigvalsh(array)
    if np.min(eigenvalues) < -1e-10:
        raise ValueError(
            f"{name} must be positive semidefinite. Minimum eigenvalue={np.min(eigenvalues)}."
        )


def validate_market_state_setup(
    T: int,
    market_state_setup: Mapping[str, object],
) -> None:
    """檢查 state sequence 或 Markov transition matrix 設定。"""

    state_sequence = market_state_setup.get("state_sequence")
    if state_sequence is not None:
        if len(state_sequence) != T:
            raise ValueError(
                "state_sequence length must equal T. "
                f"Received length={len(state_sequence)}, T={T}."
            )
        _validate_state_values(state_sequence, "state_sequence")
        return

    initial_state = int(market_state_setup["initial_state"])
    if initial_state not in STATE_VALUES:
        raise ValueError(
            f"initial_state must be one of -1, 0, 1. Received {initial_state}."
        )

    transition_matrix = np.asarray(market_state_setup["transition_matrix"], dtype=float)
    if transition_matrix.shape != (3, 3):
        raise ValueError(
            "transition_matrix must have shape (3, 3). "
            f"Received {transition_matrix.shape}."
        )
    if np.any(transition_matrix < 0):
        raise ValueError("transition_matrix must not contain negative probabilities.")
    if not np.allclose(transition_matrix.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Each row of transition_matrix must sum to 1.")


def validate_factor_setup(factor_vector_ar_setup: Mapping[str, object]) -> None:
    """檢查向量 AR(1) 因子參數。"""

    phi = np.asarray(factor_vector_ar_setup["Phi"], dtype=float)
    delta = np.asarray(factor_vector_ar_setup["Delta"], dtype=float)
    x0 = np.asarray(factor_vector_ar_setup["X0"], dtype=float)

    if phi.shape != (3, 3):
        raise ValueError(f"Phi must have shape (3, 3). Received {phi.shape}.")
    if delta.shape != (3,):
        raise ValueError(f"Delta must have shape (3,). Received {delta.shape}.")
    if x0.shape != (3,):
        raise ValueError(f"X0 must have shape (3,). Received {x0.shape}.")

    _validate_covariance_matrix("Sigma_X_bear", factor_vector_ar_setup["Sigma_X_bear"])
    _validate_covariance_matrix("Sigma_X_neutral", factor_vector_ar_setup["Sigma_X_neutral"])
    _validate_covariance_matrix("Sigma_X_bull", factor_vector_ar_setup["Sigma_X_bull"])


def validate_characteristic_setup(
    stock_ids: Sequence[str],
    characteristic_setup: Mapping[str, object],
) -> None:
    """檢查 characteristic 動態過程所需參數。"""

    use_shared = bool(characteristic_setup["use_shared_characteristic_params"])
    if use_shared:
        shared_params = characteristic_setup.get("shared_params")
        if shared_params is None:
            raise ValueError(
                "shared_params must be provided when use_shared_characteristic_params is True."
            )
        required_keys = {"Omega", "mu_C", "Lambda_C", "sigma_C", "C0"}
        missing_keys = required_keys - set(shared_params)
        if missing_keys:
            raise ValueError(f"shared_params is missing keys: {sorted(missing_keys)}.")
        _validate_positive("shared_params.sigma_C", float(shared_params["sigma_C"]))
        if abs(float(shared_params["Omega"])) >= 1.0:
            raise ValueError(
                f"shared_params.Omega must satisfy abs(Omega) < 1. Received {shared_params['Omega']}."
            )
        return

    per_stock_params = characteristic_setup.get("per_stock_params")
    if per_stock_params is None:
        raise ValueError(
            "per_stock_params must be provided when use_shared_characteristic_params is False."
        )

    required_keys = {"Omega_i", "mu_i", "Lambda_i", "sigma_C_i", "C0_i"}
    missing_keys = required_keys - set(per_stock_params)
    if missing_keys:
        raise ValueError(f"per_stock_params is missing keys: {sorted(missing_keys)}.")

    omega_values = _coerce_named_vector(per_stock_params["Omega_i"], stock_ids, "Omega_i")
    sigma_values = _coerce_named_vector(per_stock_params["sigma_C_i"], stock_ids, "sigma_C_i")
    _coerce_named_vector(per_stock_params["mu_i"], stock_ids, "mu_i")
    _coerce_named_vector(per_stock_params["Lambda_i"], stock_ids, "Lambda_i")
    _coerce_named_vector(per_stock_params["C0_i"], stock_ids, "C0_i")

    if np.any(np.abs(omega_values) >= 1.0):
        raise ValueError("All Omega_i values must satisfy abs(Omega_i) < 1.")
    if np.any(sigma_values <= 0):
        raise ValueError("All sigma_C_i values must be > 0.")


def validate_epsilon_setup(
    stock_ids: Sequence[str],
    epsilon_setup: Mapping[str, object],
) -> None:
    """檢查 epsilon 參數。"""

    if bool(epsilon_setup["use_shared_sigma_epsilon"]):
        _validate_positive("shared_sigma_epsilon", float(epsilon_setup["shared_sigma_epsilon"]))
        return

    per_stock_sigma = epsilon_setup.get("per_stock_sigma_epsilon_i")
    if per_stock_sigma is None:
        raise ValueError(
            "per_stock_sigma_epsilon_i must be provided when use_shared_sigma_epsilon is False."
        )

    sigma_values = _coerce_named_vector(
        per_stock_sigma,
        stock_ids,
        "per_stock_sigma_epsilon_i",
    )
    if np.any(sigma_values <= 0):
        raise ValueError("All per_stock_sigma_epsilon_i values must be > 0.")


def validate_clipping_price_setup(
    stock_ids: Sequence[str],
    clipping_price_setup: Mapping[str, object],
) -> None:
    """檢查 clipping 與價格設定。"""

    limit_down = float(clipping_price_setup["limit_down"])
    limit_up = float(clipping_price_setup["limit_up"])
    if limit_down >= limit_up:
        raise ValueError(
            f"limit_down must be smaller than limit_up. Received {limit_down} and {limit_up}."
        )
    if limit_down <= -1.0:
        raise ValueError(
            f"limit_down must be > -1 so prices stay non-negative. Received {limit_down}."
        )

    if bool(clipping_price_setup["shared_init_price"]):
        _validate_positive("initial_price", float(clipping_price_setup["initial_price"]))
        return

    per_stock_initial_price = clipping_price_setup.get("per_stock_initial_price")
    if per_stock_initial_price is None:
        raise ValueError(
            "per_stock_initial_price must be provided when shared_init_price is False."
        )

    initial_prices = _coerce_named_vector(
        per_stock_initial_price,
        stock_ids,
        "per_stock_initial_price",
    )
    if np.any(initial_prices <= 0):
        raise ValueError("All per_stock_initial_price values must be > 0.")


def validate_simulation_inputs(
    N: int,
    T: int,
    market_state_setup: Mapping[str, object],
    factor_vector_ar_setup: Mapping[str, object],
    characteristic_setup: Mapping[str, object],
    alpha_setup: Mapping[str, object],
    epsilon_setup: Mapping[str, object],
    clipping_price_setup: Mapping[str, object],
    stock_ids: Sequence[str],
) -> None:
    """檢查整個模擬流程的主要輸入參數。"""

    if N <= 0:
        raise ValueError(f"N must be > 0. Received {N}.")
    if T <= 0:
        raise ValueError(f"T must be > 0. Received {T}.")

    validate_market_state_setup(T=T, market_state_setup=market_state_setup)
    validate_factor_setup(factor_vector_ar_setup=factor_vector_ar_setup)
    validate_characteristic_setup(stock_ids=stock_ids, characteristic_setup=characteristic_setup)

    _validate_positive("sigma_alpha", float(alpha_setup["sigma_alpha"]))
    validate_epsilon_setup(stock_ids=stock_ids, epsilon_setup=epsilon_setup)
    validate_clipping_price_setup(
        stock_ids=stock_ids,
        clipping_price_setup=clipping_price_setup,
    )


def validate_panel_row_count(panel_df: pd.DataFrame, expected_rows: int) -> None:
    """檢查 merge 後資料筆數是否符合預期。"""

    if len(panel_df) != expected_rows:
        raise ValueError(
            "Merged panel row count does not match expectation. "
            f"Expected {expected_rows}, received {len(panel_df)}."
        )

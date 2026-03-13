"""
這個模組負責檢查主模擬流程的輸入參數是否合法。
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
    """檢查 state sequence 是否只包含 -1、0、1。"""

    invalid_values = sorted(set(int(value) for value in state_values) - set(STATE_VALUES))
    if invalid_values:
        raise ValueError(
            f"{name} must only contain -1, 0, 1. Received invalid values: {invalid_values}."
        )


def _coerce_array(values: Sequence[float] | Sequence[Sequence[float]], name: str) -> np.ndarray:
    """把輸入值轉成 NumPy array。"""

    try:
        return np.asarray(values, dtype=float)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} could not be converted to a numeric array.") from exc


def _validate_covariance_matrix(name: str, matrix: Sequence[Sequence[float]], shape: tuple[int, int]) -> None:
    """檢查 covariance matrix 的 shape、對稱性與半正定性。"""

    array = _coerce_array(matrix, name)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}. Received {array.shape}.")
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
    """檢查 state sequence 或 Markov transition matrix。"""

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

    transition_matrix = _coerce_array(market_state_setup["transition_matrix"], "transition_matrix")
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
    """檢查 3 維向量 AR(1) 因子參數。"""

    phi = _coerce_array(factor_vector_ar_setup["Phi"], "Phi")
    delta = _coerce_array(factor_vector_ar_setup["Delta"], "Delta")
    x0 = _coerce_array(factor_vector_ar_setup["X0"], "X0")

    if phi.shape != (3, 3):
        raise ValueError(f"Phi must have shape (3, 3). Received {phi.shape}.")
    if delta.shape != (3,):
        raise ValueError(f"Delta must have shape (3,). Received {delta.shape}.")
    if x0.shape != (3,):
        raise ValueError(f"X0 must have shape (3,). Received {x0.shape}.")

    _validate_covariance_matrix("Sigma_X_bear", factor_vector_ar_setup["Sigma_X_bear"], (3, 3))
    _validate_covariance_matrix(
        "Sigma_X_neutral",
        factor_vector_ar_setup["Sigma_X_neutral"],
        (3, 3),
    )
    _validate_covariance_matrix("Sigma_X_bull", factor_vector_ar_setup["Sigma_X_bull"], (3, 3))


def validate_characteristic_setup(
    N: int,
    characteristic_setup: Mapping[str, object],
) -> None:
    """檢查三維 characteristic vector 參數。"""

    use_shared = bool(characteristic_setup["use_shared_characteristic_params"])
    if use_shared:
        shared_params = characteristic_setup.get("shared_params")
        if shared_params is None:
            raise ValueError(
                "shared_params must be provided when use_shared_characteristic_params is True."
            )

        omega = _coerce_array(shared_params["Omega"], "Omega")
        mu = _coerce_array(shared_params["mu_C"], "mu_C")
        lambda_vector = _coerce_array(shared_params["Lambda_C"], "Lambda_C")
        sigma = _coerce_array(shared_params["sigma_C"], "sigma_C")
        c0 = _coerce_array(shared_params["C0"], "C0")

        if omega.shape != (3,):
            raise ValueError(f"Omega must have shape (3,). Received {omega.shape}.")
        if mu.shape != (3,):
            raise ValueError(f"mu_C must have shape (3,). Received {mu.shape}.")
        if lambda_vector.shape != (3,):
            raise ValueError(f"Lambda_C must have shape (3,). Received {lambda_vector.shape}.")
        if sigma.shape != (3,):
            raise ValueError(f"sigma_C must have shape (3,). Received {sigma.shape}.")
        if c0.shape != (3,):
            raise ValueError(f"C0 must have shape (3,). Received {c0.shape}.")
        if np.any(sigma <= 0):
            raise ValueError("Every component of sigma_C must be > 0.")
        if np.any(np.abs(omega) >= 1.0):
            raise ValueError("Every component of Omega must satisfy abs(Omega) < 1.")
        return

    per_stock_params = characteristic_setup.get("per_stock_params")
    if per_stock_params is None:
        raise ValueError(
            "per_stock_params must be provided when use_shared_characteristic_params is False."
        )

    omega = _coerce_array(per_stock_params["Omega_i"], "Omega_i")
    mu = _coerce_array(per_stock_params["mu_i"], "mu_i")
    lambda_vector = _coerce_array(per_stock_params["Lambda_i"], "Lambda_i")
    sigma = _coerce_array(per_stock_params["sigma_C_i"], "sigma_C_i")
    c0 = _coerce_array(per_stock_params["C0_i"], "C0_i")

    if omega.shape != (N, 3):
        raise ValueError(f"Omega_i must have shape (N, 3). Received {omega.shape}.")
    if mu.shape != (N, 3):
        raise ValueError(f"mu_i must have shape (N, 3). Received {mu.shape}.")
    if lambda_vector.shape != (N, 3):
        raise ValueError(f"Lambda_i must have shape (N, 3). Received {lambda_vector.shape}.")
    if sigma.shape != (N, 3):
        raise ValueError(f"sigma_C_i must have shape (N, 3). Received {sigma.shape}.")
    if c0.shape != (N, 3):
        raise ValueError(f"C0_i must have shape (N, 3). Received {c0.shape}.")
    if np.any(sigma <= 0):
        raise ValueError("Every component of sigma_C_i must be > 0.")
    if np.any(np.abs(omega) >= 1.0):
        raise ValueError("Every component of Omega_i must satisfy abs(Omega_i) < 1.")


def validate_exposure_setup(exposure_setup: Mapping[str, object]) -> None:
    """檢查 exposure loading vectors 與 scalar intercepts。"""

    for vector_name in ("a_mkt", "a_smb", "a_hml"):
        vector = _coerce_array(exposure_setup[vector_name], vector_name)
        if vector.shape != (3,):
            raise ValueError(f"{vector_name} must have shape (3,). Received {vector.shape}.")

    for scalar_name in ("b_mkt", "b_smb", "b_hml"):
        scalar_value = np.asarray(exposure_setup[scalar_name], dtype=float)
        if scalar_value.shape != ():
            raise ValueError(f"{scalar_name} must be a scalar. Received shape {scalar_value.shape}.")


def validate_epsilon_setup(
    N: int,
    epsilon_setup: Mapping[str, object],
) -> None:
    """檢查 epsilon 參數。"""

    if bool(epsilon_setup["use_shared_sigma_epsilon"]):
        _validate_positive("shared_sigma_epsilon", float(epsilon_setup["shared_sigma_epsilon"]))
        return

    per_stock_sigma = _coerce_array(
        epsilon_setup["per_stock_sigma_epsilon_i"],
        "per_stock_sigma_epsilon_i",
    )
    if per_stock_sigma.shape != (N,):
        raise ValueError(
            "per_stock_sigma_epsilon_i must have shape (N,). "
            f"Received {per_stock_sigma.shape}."
        )
    if np.any(per_stock_sigma <= 0):
        raise ValueError("All per_stock_sigma_epsilon_i values must be > 0.")


def validate_clipping_price_setup(
    N: int,
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

    per_stock_initial_price = _coerce_array(
        clipping_price_setup["per_stock_initial_price"],
        "per_stock_initial_price",
    )
    if per_stock_initial_price.shape != (N,):
        raise ValueError(
            "per_stock_initial_price must have shape (N,). "
            f"Received {per_stock_initial_price.shape}."
        )
    if np.any(per_stock_initial_price <= 0):
        raise ValueError("All per_stock_initial_price values must be > 0.")


def validate_simulation_inputs(
    N: int,
    T: int,
    market_state_setup: Mapping[str, object],
    factor_vector_ar_setup: Mapping[str, object],
    characteristic_setup: Mapping[str, object],
    exposure_setup: Mapping[str, object],
    alpha_setup: Mapping[str, object],
    epsilon_setup: Mapping[str, object],
    clipping_price_setup: Mapping[str, object],
) -> None:
    """檢查整個模擬流程的主要輸入參數。"""

    if N <= 0:
        raise ValueError(f"N must be > 0. Received {N}.")
    if T <= 0:
        raise ValueError(f"T must be > 0. Received {T}.")

    validate_market_state_setup(T=T, market_state_setup=market_state_setup)
    validate_factor_setup(factor_vector_ar_setup=factor_vector_ar_setup)
    validate_characteristic_setup(N=N, characteristic_setup=characteristic_setup)
    validate_exposure_setup(exposure_setup=exposure_setup)

    _validate_positive("sigma_alpha", float(alpha_setup["sigma_alpha"]))
    validate_epsilon_setup(N=N, epsilon_setup=epsilon_setup)
    validate_clipping_price_setup(N=N, clipping_price_setup=clipping_price_setup)


def validate_component_row_count(name: str, df: pd.DataFrame, expected_rows: int) -> None:
    """檢查中間資料表筆數是否符合預期。"""

    if len(df) != expected_rows:
        raise ValueError(
            f"{name} row count does not match expectation. "
            f"Expected {expected_rows}, received {len(df)}."
        )


def validate_panel_row_count(panel_df: pd.DataFrame, expected_rows: int) -> None:
    """檢查 merge 後資料筆數是否符合預期。"""

    if len(panel_df) != expected_rows:
        raise ValueError(
            "Merged panel row count does not match expectation. "
            f"Expected {expected_rows}, received {len(panel_df)}."
        )

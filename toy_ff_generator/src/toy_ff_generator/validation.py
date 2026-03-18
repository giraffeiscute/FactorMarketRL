from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from toy_ff_generator.characteristics import (
    FIRM_CHARACTERISTIC_COLUMNS,
    LATENT_STATE_COLUMNS,
    LATENT_STATE_DIM,
)

STATE_VALUES = (-1, 0, 1)
ALPHA_EPSILON_GROUPS = ("low", "mid", "high")
BETA_COLUMNS = ("beta_mkt", "beta_smb", "beta_hml")


def _latent_state_shape_message(expected_shape: tuple[int, ...]) -> str:
    return (
        f"must have shape {expected_shape} for the latent state order "
        f"{LATENT_STATE_COLUMNS}"
    )


def _observable_characteristic_shape_message(expected_shape: tuple[int, ...]) -> str:
    return (
        f"must have shape {expected_shape} for the observable characteristic order "
        f"{FIRM_CHARACTERISTIC_COLUMNS}"
    )


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0. Received {value}.")


def _validate_required_columns(
    df: pd.DataFrame,
    *,
    name: str,
    required_columns: Sequence[str],
) -> None:
    missing_columns = [
        column_name for column_name in required_columns if column_name not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{name} is missing required columns {missing_columns}. "
            f"Expected at least {list(required_columns)}."
        )


def _validate_required_columns_not_null(
    df: pd.DataFrame,
    *,
    name: str,
    required_columns: Sequence[str],
) -> None:
    if df[list(required_columns)].isna().any().any():
        raise ValueError(f"{name} must not contain null values in {list(required_columns)}.")


def _validate_state_values(state_values: Sequence[int], name: str) -> None:
    invalid_values = sorted(set(int(value) for value in state_values) - set(STATE_VALUES))
    if invalid_values:
        raise ValueError(
            f"{name} must only contain -1, 0, 1. Received invalid values: {invalid_values}."
        )


def _coerce_array(values: Sequence[float] | Sequence[Sequence[float]], name: str) -> np.ndarray:
    try:
        return np.asarray(values, dtype=float)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"{name} could not be converted to a numeric array.") from exc


def _validate_latent_state_array_shape(
    name: str,
    array: np.ndarray,
    expected_shape: tuple[int, ...],
) -> None:
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} {_latent_state_shape_message(expected_shape)}. Received {array.shape}."
        )


def _validate_observable_array_shape(
    name: str,
    array: np.ndarray,
    expected_shape: tuple[int, ...],
) -> None:
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} {_observable_characteristic_shape_message(expected_shape)}. "
            f"Received {array.shape}."
        )


def _validate_covariance_matrix(
    name: str,
    matrix: Sequence[Sequence[float]],
    shape: tuple[int, int],
) -> None:
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
    phi = _coerce_array(factor_vector_ar_setup["Phi"], "Phi")
    x0 = _coerce_array(factor_vector_ar_setup["X0"], "X0")
    mu_bear = factor_vector_ar_setup.get("mu_bear")
    mu_neutral = factor_vector_ar_setup.get("mu_neutral")
    mu_bull = factor_vector_ar_setup.get("mu_bull")

    if phi.shape != (3, 3):
        raise ValueError(f"Phi must have shape (3, 3). Received {phi.shape}.")
    if x0.shape != (3,):
        raise ValueError(f"X0 must have shape (3,). Received {x0.shape}.")

    has_explicit_means = any(value is not None for value in (mu_bear, mu_neutral, mu_bull))
    if has_explicit_means:
        if any(value is None for value in (mu_bear, mu_neutral, mu_bull)):
            raise ValueError(
                "mu_bear, mu_neutral, and mu_bull must all be provided together."
            )
        for name, value in (
            ("mu_bear", mu_bear),
            ("mu_neutral", mu_neutral),
            ("mu_bull", mu_bull),
        ):
            vector = _coerce_array(value, name)
            if vector.shape != (3,):
                raise ValueError(f"{name} must have shape (3,). Received {vector.shape}.")
        if factor_vector_ar_setup.get("Delta") is not None:
            warnings.warn(
                "Delta is deprecated and ignored when mu_bear/mu_neutral/mu_bull are provided.",
                DeprecationWarning,
                stacklevel=2,
            )
    else:
        delta = factor_vector_ar_setup.get("Delta")
        if delta is None:
            raise ValueError(
                "factor_vector_ar_setup must provide either mu_bear/mu_neutral/mu_bull or deprecated Delta."
            )
        delta_vector = _coerce_array(delta, "Delta")
        if delta_vector.shape != (3,):
            raise ValueError(f"Delta must have shape (3,). Received {delta_vector.shape}.")
        warnings.warn(
            "Delta is deprecated. Use mu_bear/mu_neutral/mu_bull instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    _validate_covariance_matrix("Sigma_X_bear", factor_vector_ar_setup["Sigma_X_bear"], (3, 3))
    _validate_covariance_matrix(
        "Sigma_X_neutral",
        factor_vector_ar_setup["Sigma_X_neutral"],
        (3, 3),
    )
    _validate_covariance_matrix("Sigma_X_bull", factor_vector_ar_setup["Sigma_X_bull"], (3, 3))


def validate_beta_class_setup(beta_class_setup: Mapping[str, object]) -> None:
    class_centers = beta_class_setup.get("class_centers")
    if not isinstance(class_centers, Mapping):
        raise ValueError("beta_class_setup['class_centers'] must be a mapping.")

    missing_groups = [
        group_name for group_name in ALPHA_EPSILON_GROUPS if group_name not in class_centers
    ]
    if missing_groups:
        raise ValueError(
            f"class_centers must define {list(ALPHA_EPSILON_GROUPS)}. Missing {missing_groups}."
        )

    for group_name in ALPHA_EPSILON_GROUPS:
        scalar_value = np.asarray(class_centers[group_name], dtype=float)
        if scalar_value.shape != ():
            raise ValueError(
                f"class_centers[{group_name!r}] must be a scalar. Received shape {scalar_value.shape}."
            )


def validate_latent_characteristic_setup(
    N: int,
    latent_characteristic_setup: Mapping[str, object],
) -> None:
    use_shared = bool(latent_characteristic_setup["use_shared_latent_state_params"])
    if use_shared:
        shared_params = latent_characteristic_setup.get("shared_params")
        if shared_params is None:
            raise ValueError(
                "shared_params must be provided when use_shared_latent_state_params is True."
            )

        omega = _coerce_array(shared_params["Omega"], "Omega")
        mu = _coerce_array(shared_params["mu_Z"], "mu_Z")
        lambda_vector = _coerce_array(shared_params["lambda_Z"], "lambda_Z")
        sigma = _coerce_array(shared_params["sigma_Z"], "sigma_Z")
        z0 = _coerce_array(shared_params["Z0"], "Z0")

        _validate_latent_state_array_shape("Omega", omega, (LATENT_STATE_DIM,))
        _validate_latent_state_array_shape("mu_Z", mu, (LATENT_STATE_DIM,))
        _validate_latent_state_array_shape("lambda_Z", lambda_vector, (LATENT_STATE_DIM,))
        _validate_latent_state_array_shape("sigma_Z", sigma, (LATENT_STATE_DIM,))
        _validate_latent_state_array_shape("Z0", z0, (LATENT_STATE_DIM,))
        if np.any(sigma <= 0):
            raise ValueError("Every component of sigma_Z must be > 0.")
        if np.any(np.abs(omega) >= 1.0):
            raise ValueError("Every component of Omega must satisfy abs(Omega) < 1.")
        return

    per_stock_params = latent_characteristic_setup.get("per_stock_params")
    if per_stock_params is None:
        raise ValueError(
            "per_stock_params must be provided when use_shared_latent_state_params is False."
        )

    omega = _coerce_array(per_stock_params["Omega_i"], "Omega_i")
    mu = _coerce_array(per_stock_params["mu_i"], "mu_i")
    lambda_vector = _coerce_array(per_stock_params["lambda_i"], "lambda_i")
    sigma = _coerce_array(per_stock_params["sigma_Z_i"], "sigma_Z_i")
    z0 = _coerce_array(per_stock_params["Z0_i"], "Z0_i")

    latent_matrix_shape = (N, LATENT_STATE_DIM)
    _validate_latent_state_array_shape("Omega_i", omega, latent_matrix_shape)
    _validate_latent_state_array_shape("mu_i", mu, latent_matrix_shape)
    _validate_latent_state_array_shape("lambda_i", lambda_vector, latent_matrix_shape)
    _validate_latent_state_array_shape("sigma_Z_i", sigma, latent_matrix_shape)
    _validate_latent_state_array_shape("Z0_i", z0, latent_matrix_shape)
    if np.any(sigma <= 0):
        raise ValueError("Every component of sigma_Z_i must be > 0.")
    if np.any(np.abs(omega) >= 1.0):
        raise ValueError("Every component of Omega_i must satisfy abs(Omega_i) < 1.")


def validate_exposure_setup(exposure_setup: Mapping[str, object]) -> None:
    exposure_matrix = _coerce_array(exposure_setup["A"], "A")
    intercept_vector = _coerce_array(exposure_setup["b"], "b")
    if exposure_matrix.shape != (LATENT_STATE_DIM, LATENT_STATE_DIM):
        raise ValueError(
            f"A must have shape ({LATENT_STATE_DIM}, {LATENT_STATE_DIM}). Received {exposure_matrix.shape}."
        )
    if intercept_vector.shape != (LATENT_STATE_DIM,):
        raise ValueError(
            f"b must have shape ({LATENT_STATE_DIM},). Received {intercept_vector.shape}."
        )


def validate_latent_state_df(
    latent_state_df: pd.DataFrame,
    expected_rows: int,
) -> None:
    if len(latent_state_df) != expected_rows:
        raise ValueError(
            "latent_state_df row count does not match expectation. "
            f"Expected {expected_rows}, received {len(latent_state_df)}."
        )

    missing_columns = [
        column_name
        for column_name in LATENT_STATE_COLUMNS
        if column_name not in latent_state_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "latent_state_df is missing required latent state columns "
            f"{missing_columns}. Expected {LATENT_STATE_COLUMNS}."
        )

    latent_state_matrix = latent_state_df[LATENT_STATE_COLUMNS].to_numpy(dtype=float)
    _validate_latent_state_array_shape(
        "latent_state_df latent block",
        latent_state_matrix,
        (expected_rows, LATENT_STATE_DIM),
    )


def validate_firm_characteristics_df(
    firm_characteristics_df: pd.DataFrame,
    expected_rows: int,
) -> None:
    if len(firm_characteristics_df) != expected_rows:
        raise ValueError(
            "firm_characteristics_df row count does not match expectation. "
            f"Expected {expected_rows}, received {len(firm_characteristics_df)}."
        )

    missing_columns = [
        column_name
        for column_name in FIRM_CHARACTERISTIC_COLUMNS
        if column_name not in firm_characteristics_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "firm_characteristics_df is missing required observable columns "
            f"{missing_columns}. Expected {FIRM_CHARACTERISTIC_COLUMNS}."
        )

    observable_matrix = firm_characteristics_df[FIRM_CHARACTERISTIC_COLUMNS].to_numpy(dtype=float)
    _validate_observable_array_shape(
        "firm_characteristics_df observable block",
        observable_matrix,
        (expected_rows, LATENT_STATE_DIM),
    )


def _validate_group_levels(
    levels: Mapping[str, object],
    *,
    levels_name: str,
) -> None:
    missing_groups = [
        group_name for group_name in ALPHA_EPSILON_GROUPS if group_name not in levels
    ]
    if missing_groups:
        raise ValueError(
            f"{levels_name} must define {list(ALPHA_EPSILON_GROUPS)}. "
            f"Missing groups: {missing_groups}."
        )

    for group_name in ALPHA_EPSILON_GROUPS:
        scalar_value = np.asarray(levels[group_name], dtype=float)
        if scalar_value.shape != ():
            raise ValueError(
                f"{levels_name}[{group_name!r}] must be a scalar. "
                f"Received shape {scalar_value.shape}."
            )


def validate_alpha_epsilon_mode_setup(
    alpha_epsilon_mode_setup: Mapping[str, object],
) -> None:
    alpha_group = str(alpha_epsilon_mode_setup["alpha_group"])
    epsilon_group = str(alpha_epsilon_mode_setup["epsilon_group"])
    alpha_levels = alpha_epsilon_mode_setup["alpha_levels"]
    epsilon_levels = alpha_epsilon_mode_setup["epsilon_levels"]

    if not isinstance(alpha_levels, Mapping):
        raise ValueError("alpha_levels must be a mapping from group name to alpha value.")
    if not isinstance(epsilon_levels, Mapping):
        raise ValueError("epsilon_levels must be a mapping from group name to epsilon sigma.")

    _validate_group_levels(alpha_levels, levels_name="alpha_levels")
    _validate_group_levels(epsilon_levels, levels_name="epsilon_levels")

    if alpha_group not in alpha_levels:
        raise ValueError(
            f"alpha_group must be one of {sorted(alpha_levels)}. Received {alpha_group!r}."
        )
    if epsilon_group not in epsilon_levels:
        raise ValueError(
            f"epsilon_group must be one of {sorted(epsilon_levels)}. Received {epsilon_group!r}."
        )

    for group_name in ALPHA_EPSILON_GROUPS:
        _validate_positive(
            f"epsilon_levels[{group_name!r}]",
            float(epsilon_levels[group_name]),
        )


def validate_clipping_price_setup(
    N: int,
    clipping_price_setup: Mapping[str, object],
) -> None:
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
    beta_class_setup: Mapping[str, object],
    latent_characteristic_setup: Mapping[str, object],
    exposure_setup: Mapping[str, object],
    alpha_epsilon_mode_setup: Mapping[str, object],
    clipping_price_setup: Mapping[str, object],
) -> None:
    if N <= 0:
        raise ValueError(f"N must be > 0. Received {N}.")
    if T <= 0:
        raise ValueError(f"T must be > 0. Received {T}.")

    validate_market_state_setup(T=T, market_state_setup=market_state_setup)
    validate_factor_setup(factor_vector_ar_setup=factor_vector_ar_setup)
    validate_beta_class_setup(beta_class_setup=beta_class_setup)
    validate_latent_characteristic_setup(N=N, latent_characteristic_setup=latent_characteristic_setup)
    validate_exposure_setup(exposure_setup=exposure_setup)
    validate_alpha_epsilon_mode_setup(alpha_epsilon_mode_setup=alpha_epsilon_mode_setup)
    validate_clipping_price_setup(N=N, clipping_price_setup=clipping_price_setup)


def validate_component_row_count(name: str, df: pd.DataFrame, expected_rows: int) -> None:
    if len(df) != expected_rows:
        raise ValueError(
            f"{name} row count does not match expectation. "
            f"Expected {expected_rows}, received {len(df)}."
        )


def validate_panel_row_count(panel_df: pd.DataFrame, expected_rows: int) -> None:
    if len(panel_df) != expected_rows:
        raise ValueError(
            "Merged panel row count does not match expectation. "
            f"Expected {expected_rows}, received {len(panel_df)}."
        )


def validate_beta_df(
    beta_df: pd.DataFrame,
    expected_rows: int | None = None,
) -> None:
    required_columns = ("stock_id", "t", *BETA_COLUMNS)
    if beta_df.empty:
        raise ValueError("beta_df must not be empty.")

    _validate_required_columns(
        beta_df,
        name="beta_df",
        required_columns=required_columns,
    )
    _validate_required_columns_not_null(
        beta_df,
        name="beta_df",
        required_columns=required_columns,
    )

    if expected_rows is not None:
        validate_component_row_count(
            name="beta_df",
            df=beta_df,
            expected_rows=expected_rows,
        )


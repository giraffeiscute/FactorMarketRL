"""
這個模組負責生成 latent characteristic state，並映射成可觀察的 firm characteristics。

內部 latent state 順序固定為：
- X[:, 0] = latent_size_state
- X[:, 1] = latent_book_to_price_state

對外可觀察 firm characteristics 順序固定為：
- firm_size = exp(latent_size_state)
- book_to_price = exp(latent_book_to_price_state)
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

LATENT_STATE_NAMES = ("latent_size_state", "latent_book_to_price_state")
FIRM_CHARACTERISTIC_NAMES = ("firm_size", "book_to_price")
LATENT_STATE_COLUMNS = list(LATENT_STATE_NAMES)
FIRM_CHARACTERISTIC_COLUMNS = list(FIRM_CHARACTERISTIC_NAMES)
LATENT_STATE_DIM = len(LATENT_STATE_COLUMNS)


def _shared_vector_to_named_columns(prefix: str, names: Sequence[str], vector: np.ndarray) -> dict[str, float]:
    return {f"{prefix}_{name}": float(vector[idx]) for idx, name in enumerate(names)}


def _matrix_to_named_columns(prefix: str, names: Sequence[str], matrix: np.ndarray) -> dict[str, np.ndarray]:
    return {f"{prefix}_{name}": matrix[:, idx] for idx, name in enumerate(names)}


def _row_vector(row: object, prefix: str, names: Sequence[str]) -> np.ndarray:
    return np.asarray([getattr(row, f"{prefix}_{name}") for name in names], dtype=float)


def _coerce_shared_latent_vector(
    shared_params: Mapping[str, Sequence[float]],
    key: str,
) -> np.ndarray:
    """把 shared latent state 參數轉成長度 2 向量。"""

    vector = np.asarray(shared_params[key], dtype=float)
    if vector.shape != (LATENT_STATE_DIM,):
        raise ValueError(
            f"{key} must have shape ({LATENT_STATE_DIM},) for "
            f"{list(LATENT_STATE_COLUMNS)}. Received {vector.shape}."
        )
    return vector


def _coerce_per_stock_latent_matrix(
    per_stock_params: Mapping[str, Sequence[Sequence[float]]],
    key: str,
) -> np.ndarray:
    """把 per-stock latent state 參數轉成 shape (N, 2) 的矩陣。"""

    matrix = np.asarray(per_stock_params[key], dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != LATENT_STATE_DIM:
        raise ValueError(
            f"{key} must have shape (N, {LATENT_STATE_DIM}) for "
            f"{list(LATENT_STATE_COLUMNS)}. Received {matrix.shape}."
        )
    return matrix


def _build_latent_state_param_table(
    stock_ids: Sequence[str],
    use_shared_latent_state_params: bool,
    shared_params: Mapping[str, Sequence[float]] | None,
    per_stock_params: Mapping[str, Sequence[Sequence[float]]] | None,
) -> pd.DataFrame:
    """建立每支股票對應的 latent state 參數表。"""

    stock_count = len(stock_ids)

    if use_shared_latent_state_params:
        if shared_params is None:
            raise ValueError(
                "shared_params is required when use_shared_latent_state_params is True."
            )

        omega = _coerce_shared_latent_vector(shared_params, "Omega")
        mu = _coerce_shared_latent_vector(shared_params, "mu_X")
        lambda_vector = _coerce_shared_latent_vector(shared_params, "lambda_X")
        sigma = _coerce_shared_latent_vector(shared_params, "sigma_X")
        x0 = _coerce_shared_latent_vector(shared_params, "X0")

        return pd.DataFrame(
            {
                "stock_id": list(stock_ids),
                **_shared_vector_to_named_columns("Omega", LATENT_STATE_NAMES, omega),
                **_shared_vector_to_named_columns("mu", LATENT_STATE_NAMES, mu),
                **_shared_vector_to_named_columns("lambda", LATENT_STATE_NAMES, lambda_vector),
                **_shared_vector_to_named_columns("sigma_X", LATENT_STATE_NAMES, sigma),
                **_shared_vector_to_named_columns("X0", LATENT_STATE_NAMES, x0),
            }
        )

    if per_stock_params is None:
        raise ValueError(
            "per_stock_params is required when use_shared_latent_state_params is False."
        )

    omega = _coerce_per_stock_latent_matrix(per_stock_params, "Omega_i")
    mu = _coerce_per_stock_latent_matrix(per_stock_params, "mu_i")
    lambda_vector = _coerce_per_stock_latent_matrix(per_stock_params, "lambda_i")
    sigma = _coerce_per_stock_latent_matrix(per_stock_params, "sigma_X_i")
    x0 = _coerce_per_stock_latent_matrix(per_stock_params, "X0_i")

    for name, matrix in (
        ("Omega_i", omega),
        ("mu_i", mu),
        ("lambda_i", lambda_vector),
        ("sigma_X_i", sigma),
        ("X0_i", x0),
    ):
        if matrix.shape[0] != stock_count:
            raise ValueError(
                f"{name} must have {stock_count} rows to match stock_ids. Received {matrix.shape[0]}."
            )

    return pd.DataFrame(
        {
            "stock_id": list(stock_ids),
            **_matrix_to_named_columns("Omega", LATENT_STATE_NAMES, omega),
            **_matrix_to_named_columns("mu", LATENT_STATE_NAMES, mu),
            **_matrix_to_named_columns("lambda", LATENT_STATE_NAMES, lambda_vector),
            **_matrix_to_named_columns("sigma_X", LATENT_STATE_NAMES, sigma),
            **_matrix_to_named_columns("X0", LATENT_STATE_NAMES, x0),
        }
    )


def latent_to_firm_characteristics(latent_state_values: np.ndarray) -> np.ndarray:
    """把 latent state 映射成恆為正的 observable firm characteristics。"""

    latent_state_array = np.asarray(latent_state_values, dtype=float)
    if latent_state_array.shape[-1] != LATENT_STATE_DIM:
        raise ValueError(
            f"latent_state_values must have trailing dimension {LATENT_STATE_DIM}. "
            f"Received {latent_state_array.shape}."
        )
    return np.exp(latent_state_array)


def state_to_firm_characteristics(
    latent_state_df: pd.DataFrame,
) -> pd.DataFrame:
    """把 latent state DataFrame 轉成 observable firm characteristics DataFrame。"""

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

    observable_values = latent_to_firm_characteristics(
        latent_state_df[LATENT_STATE_COLUMNS].to_numpy(dtype=float)
    )
    firm_characteristics_df = latent_state_df[["stock_id", "t"]].copy()
    for idx, column_name in enumerate(FIRM_CHARACTERISTIC_COLUMNS):
        firm_characteristics_df[column_name] = observable_values[:, idx]

    return firm_characteristics_df


def generate_latent_characteristic_states(
    stock_ids: Sequence[str],
    time_columns: Sequence[str],
    state_sequence: Sequence[int],
    use_shared_latent_state_params: bool,
    rng: np.random.Generator,
    shared_params: Mapping[str, Sequence[float]] | None = None,
    per_stock_params: Mapping[str, Sequence[Sequence[float]]] | None = None,
) -> pd.DataFrame:
    """生成 latent state DataFrame，欄位為 `[stock_id, t, latent_size_state, latent_book_to_price_state]`。"""

    param_df = _build_latent_state_param_table(
        stock_ids=stock_ids,
        use_shared_latent_state_params=use_shared_latent_state_params,
        shared_params=shared_params,
        per_stock_params=per_stock_params,
    )

    rows: list[dict[str, float | str]] = []
    for row in param_df.itertuples(index=False):
        previous = _row_vector(row, "X0", LATENT_STATE_NAMES)
        omega = _row_vector(row, "Omega", LATENT_STATE_NAMES)
        mu = _row_vector(row, "mu", LATENT_STATE_NAMES)
        lambda_vector = _row_vector(row, "lambda", LATENT_STATE_NAMES)
        sigma = _row_vector(row, "sigma_X", LATENT_STATE_NAMES)

        for time_label, state in zip(time_columns, state_sequence, strict=True):
            innovation = rng.normal(loc=0.0, scale=sigma, size=LATENT_STATE_DIM)
            current = omega * previous + mu + lambda_vector * state + innovation
            rows.append(
                {
                    "stock_id": row.stock_id,
                    "t": time_label,
                    **{
                        column_name: float(current[idx])
                        for idx, column_name in enumerate(LATENT_STATE_COLUMNS)
                    },
                }
            )
            previous = current

    return pd.DataFrame(rows)

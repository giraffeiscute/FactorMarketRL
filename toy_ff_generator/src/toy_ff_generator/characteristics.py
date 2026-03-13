"""
這個模組負責生成三維動態 characteristic vector。

對每支股票 i、每個時間 t，characteristic 由下式遞迴生成：

C_{i,t}^{(d)} = Omega_i^{(d)} * C_{i,t-1}^{(d)}
              + mu_i^{(d)}
              + Lambda_i^{(d)} * S_t
              + xi_{i,t}^{(d)}

其中 d = 1, 2, 3，且 xi 的三個分量彼此獨立。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

CHARACTERISTIC_COLUMNS = ["C1", "C2", "C3"]


def _coerce_shared_vector(shared_params: Mapping[str, Sequence[float]], key: str) -> np.ndarray:
    """把 shared characteristic 參數轉成長度 3 向量。"""

    return np.asarray(shared_params[key], dtype=float)


def _coerce_per_stock_matrix(
    per_stock_params: Mapping[str, Sequence[Sequence[float]]],
    key: str,
) -> np.ndarray:
    """把 per-stock characteristic 參數轉成 shape (N, 3) 的矩陣。"""

    return np.asarray(per_stock_params[key], dtype=float)


def _build_characteristic_param_table(
    stock_ids: Sequence[str],
    use_shared_characteristic_params: bool,
    shared_params: Mapping[str, Sequence[float]] | None,
    per_stock_params: Mapping[str, Sequence[Sequence[float]]] | None,
) -> pd.DataFrame:
    """建立每支股票對應的 characteristic 參數表。"""

    if use_shared_characteristic_params:
        if shared_params is None:
            raise ValueError(
                "shared_params is required when use_shared_characteristic_params is True."
            )

        omega = _coerce_shared_vector(shared_params, "Omega")
        mu = _coerce_shared_vector(shared_params, "mu_C")
        lambda_vector = _coerce_shared_vector(shared_params, "Lambda_C")
        sigma = _coerce_shared_vector(shared_params, "sigma_C")
        c0 = _coerce_shared_vector(shared_params, "C0")

        return pd.DataFrame(
            {
                "stock_id": list(stock_ids),
                "Omega_1": omega[0],
                "Omega_2": omega[1],
                "Omega_3": omega[2],
                "mu_1": mu[0],
                "mu_2": mu[1],
                "mu_3": mu[2],
                "Lambda_1": lambda_vector[0],
                "Lambda_2": lambda_vector[1],
                "Lambda_3": lambda_vector[2],
                "sigma_C_1": sigma[0],
                "sigma_C_2": sigma[1],
                "sigma_C_3": sigma[2],
                "C0_1": c0[0],
                "C0_2": c0[1],
                "C0_3": c0[2],
            }
        )

    if per_stock_params is None:
        raise ValueError(
            "per_stock_params is required when use_shared_characteristic_params is False."
        )

    omega = _coerce_per_stock_matrix(per_stock_params, "Omega_i")
    mu = _coerce_per_stock_matrix(per_stock_params, "mu_i")
    lambda_vector = _coerce_per_stock_matrix(per_stock_params, "Lambda_i")
    sigma = _coerce_per_stock_matrix(per_stock_params, "sigma_C_i")
    c0 = _coerce_per_stock_matrix(per_stock_params, "C0_i")

    return pd.DataFrame(
        {
            "stock_id": list(stock_ids),
            "Omega_1": omega[:, 0],
            "Omega_2": omega[:, 1],
            "Omega_3": omega[:, 2],
            "mu_1": mu[:, 0],
            "mu_2": mu[:, 1],
            "mu_3": mu[:, 2],
            "Lambda_1": lambda_vector[:, 0],
            "Lambda_2": lambda_vector[:, 1],
            "Lambda_3": lambda_vector[:, 2],
            "sigma_C_1": sigma[:, 0],
            "sigma_C_2": sigma[:, 1],
            "sigma_C_3": sigma[:, 2],
            "C0_1": c0[:, 0],
            "C0_2": c0[:, 1],
            "C0_3": c0[:, 2],
        }
    )


def generate_characteristics(
    stock_ids: Sequence[str],
    time_columns: Sequence[str],
    state_sequence: Sequence[int],
    use_shared_characteristic_params: bool,
    rng: np.random.Generator,
    shared_params: Mapping[str, Sequence[float]] | None = None,
    per_stock_params: Mapping[str, Sequence[Sequence[float]]] | None = None,
) -> pd.DataFrame:
    """生成 characteristic_df，欄位為 `[stock_id, t, C1, C2, C3]`。"""

    param_df = _build_characteristic_param_table(
        stock_ids=stock_ids,
        use_shared_characteristic_params=use_shared_characteristic_params,
        shared_params=shared_params,
        per_stock_params=per_stock_params,
    )

    rows: list[dict[str, float | str]] = []
    for row in param_df.itertuples(index=False):
        previous = np.asarray([row.C0_1, row.C0_2, row.C0_3], dtype=float)
        omega = np.asarray([row.Omega_1, row.Omega_2, row.Omega_3], dtype=float)
        mu = np.asarray([row.mu_1, row.mu_2, row.mu_3], dtype=float)
        lambda_vector = np.asarray(
            [row.Lambda_1, row.Lambda_2, row.Lambda_3],
            dtype=float,
        )
        sigma = np.asarray([row.sigma_C_1, row.sigma_C_2, row.sigma_C_3], dtype=float)

        for time_label, state in zip(time_columns, state_sequence, strict=True):
            innovation = rng.normal(loc=0.0, scale=sigma, size=3)
            current = omega * previous + mu + lambda_vector * state + innovation
            rows.append(
                {
                    "stock_id": row.stock_id,
                    "t": time_label,
                    "C1": float(current[0]),
                    "C2": float(current[1]),
                    "C3": float(current[2]),
                }
            )
            previous = current

    return pd.DataFrame(rows)

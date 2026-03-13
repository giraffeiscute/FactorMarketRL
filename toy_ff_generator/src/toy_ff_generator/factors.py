"""
這個模組負責生成 FF 三因子時間序列。

新版模型不再把 MKT、SMB、HML 視為互相獨立的 AR(1)，
而是使用同一個 3 維向量 AR(1) 系統共同生成。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from toy_ff_generator.utils import make_time_columns


def _select_covariance_matrix(
    state: int,
    sigma_x_bear: np.ndarray,
    sigma_x_neutral: np.ndarray,
    sigma_x_bull: np.ndarray,
) -> np.ndarray:
    """根據 regime state 選擇對應的 factor covariance matrix。"""

    if state == -1:
        return sigma_x_bear
    if state == 0:
        return sigma_x_neutral
    if state == 1:
        return sigma_x_bull
    raise ValueError(f"State must be one of -1, 0, 1. Received {state}.")


def generate_factors(
    t_count: int,
    state_sequence: Sequence[int],
    X0: Sequence[float],
    Phi: Sequence[Sequence[float]],
    Delta: Sequence[float],
    Sigma_X_bear: Sequence[Sequence[float]],
    Sigma_X_neutral: Sequence[Sequence[float]],
    Sigma_X_bull: Sequence[Sequence[float]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """生成 3 維向量 AR(1) 的 factor panel。"""

    phi_matrix = np.asarray(Phi, dtype=float)
    delta_vector = np.asarray(Delta, dtype=float)
    sigma_x_bear = np.asarray(Sigma_X_bear, dtype=float)
    sigma_x_neutral = np.asarray(Sigma_X_neutral, dtype=float)
    sigma_x_bull = np.asarray(Sigma_X_bull, dtype=float)

    previous = np.asarray(X0, dtype=float)
    time_labels = make_time_columns(t_count)
    rows: list[dict[str, float | str]] = []

    for time_label, state in zip(time_labels, state_sequence, strict=True):
        covariance = _select_covariance_matrix(
            state=state,
            sigma_x_bear=sigma_x_bear,
            sigma_x_neutral=sigma_x_neutral,
            sigma_x_bull=sigma_x_bull,
        )
        shock = rng.multivariate_normal(mean=np.zeros(3, dtype=float), cov=covariance)
        current = phi_matrix @ previous + delta_vector * state + shock

        rows.append(
            {
                "t": time_label,
                "MKT": float(current[0]),
                "SMB": float(current[1]),
                "HML": float(current[2]),
            }
        )
        previous = current

    return pd.DataFrame(rows)

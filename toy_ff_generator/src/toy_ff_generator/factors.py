"""
這個模組負責生成 FF 三因子時間序列。

新版模型不再把 MKT、SMB、HML 視為互相獨立的 AR(1)，
而是使用同一個 3 維向量 AR(1) 系統共同生成。
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from toy_ff_generator.utils import make_time_columns

STATE_TO_REGIME = {-1: "bear", 0: "neutral", 1: "bull"}


def _state_to_regime_name(state: int) -> str:
    """Map numeric market state to the corresponding regime label."""

    try:
        return STATE_TO_REGIME[state]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"State must be one of -1, 0, 1. Received {state}.") from exc


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


def _resolve_regime_mean_vectors(
    mu_bear: Sequence[float] | None,
    mu_neutral: Sequence[float] | None,
    mu_bull: Sequence[float] | None,
    delta: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve factor drifts from explicit regime means or deprecated Delta."""

    has_explicit_means = any(value is not None for value in (mu_bear, mu_neutral, mu_bull))
    if has_explicit_means:
        if None in (mu_bear, mu_neutral, mu_bull):
            raise ValueError(
                "mu_bear, mu_neutral, and mu_bull must all be provided together."
            )
        if delta is not None:
            warnings.warn(
                "Delta is deprecated and ignored when mu_bear/mu_neutral/mu_bull are provided.",
                DeprecationWarning,
                stacklevel=2,
            )
        return (
            np.asarray(mu_bear, dtype=float),
            np.asarray(mu_neutral, dtype=float),
            np.asarray(mu_bull, dtype=float),
        )

    if delta is None:
        raise ValueError(
            "Either explicit regime means (mu_bear, mu_neutral, mu_bull) or deprecated Delta must be provided."
        )

    delta_vector = np.asarray(delta, dtype=float)
    warnings.warn(
        "Delta is deprecated. Use mu_bear/mu_neutral/mu_bull instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    zero_vector = np.zeros_like(delta_vector)
    return (-delta_vector, zero_vector, delta_vector)


def _select_regime_mean_vector(
    state: int,
    mu_bear: np.ndarray,
    mu_neutral: np.ndarray,
    mu_bull: np.ndarray,
) -> np.ndarray:
    """Select the factor mean vector for the given regime state."""

    regime_name = _state_to_regime_name(state)
    if regime_name == "bear":
        return mu_bear
    if regime_name == "neutral":
        return mu_neutral
    return mu_bull


def generate_factors(
    t_count: int,
    state_sequence: Sequence[int],
    X0: Sequence[float],
    Phi: Sequence[Sequence[float]],
    Delta: Sequence[float] | None,
    Sigma_X_bear: Sequence[Sequence[float]],
    Sigma_X_neutral: Sequence[Sequence[float]],
    Sigma_X_bull: Sequence[Sequence[float]],
    rng: np.random.Generator,
    mu_bear: Sequence[float] | None = None,
    mu_neutral: Sequence[float] | None = None,
    mu_bull: Sequence[float] | None = None,
) -> pd.DataFrame:
    """生成 3 維向量 AR(1) 的 factor panel。"""

    phi_matrix = np.asarray(Phi, dtype=float)
    sigma_x_bear = np.asarray(Sigma_X_bear, dtype=float)
    sigma_x_neutral = np.asarray(Sigma_X_neutral, dtype=float)
    sigma_x_bull = np.asarray(Sigma_X_bull, dtype=float)
    mu_bear_vector, mu_neutral_vector, mu_bull_vector = _resolve_regime_mean_vectors(
        mu_bear=mu_bear,
        mu_neutral=mu_neutral,
        mu_bull=mu_bull,
        delta=Delta,
    )

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
        regime_mean = _select_regime_mean_vector(
            state=state,
            mu_bear=mu_bear_vector,
            mu_neutral=mu_neutral_vector,
            mu_bull=mu_bull_vector,
        )
        shock = rng.multivariate_normal(mean=np.zeros(3, dtype=float), cov=covariance)
        current = phi_matrix @ previous + regime_mean + shock

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

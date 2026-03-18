"""Map latent characteristic states to realized FF-style beta exposures."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from toy_ff_generator.characteristics import LATENT_STATE_COLUMNS, LATENT_STATE_DIM


def _coerce_exposure_matrix(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (LATENT_STATE_DIM, LATENT_STATE_DIM):
        raise ValueError(
            f"{name} must have shape ({LATENT_STATE_DIM}, {LATENT_STATE_DIM}) for "
            f"{LATENT_STATE_COLUMNS}. Received {matrix.shape}."
        )
    return matrix


def _coerce_exposure_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (LATENT_STATE_DIM,):
        raise ValueError(
            f"{name} must have shape ({LATENT_STATE_DIM},) for "
            f"{LATENT_STATE_COLUMNS}. Received {vector.shape}."
        )
    return vector


def generate_exposures(
    latent_state_df: pd.DataFrame,
    A: Sequence[Sequence[float]],
    b: Sequence[float],
) -> pd.DataFrame:
    """Apply beta_t = A @ Z_t + b with rows ordered as MKT / SMB / HML."""

    missing_columns = [
        column_name
        for column_name in ("stock_id", "t", *LATENT_STATE_COLUMNS)
        if column_name not in latent_state_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "latent_state_df is missing required columns "
            f"{missing_columns}. Expected ['stock_id', 't', *LATENT_STATE_COLUMNS]."
        )

    latent_state_matrix = latent_state_df[LATENT_STATE_COLUMNS].to_numpy(dtype=float)
    exposure_matrix = _coerce_exposure_matrix(A, "A")
    intercept_vector = _coerce_exposure_vector(b, "b")
    beta_matrix = latent_state_matrix @ exposure_matrix.T + intercept_vector

    beta_df = latent_state_df[["stock_id", "t"]].copy()
    beta_df["beta_mkt"] = beta_matrix[:, 0]
    beta_df["beta_smb"] = beta_matrix[:, 1]
    beta_df["beta_hml"] = beta_matrix[:, 2]
    return beta_df

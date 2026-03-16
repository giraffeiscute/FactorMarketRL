"""
這個模組負責把二維 latent state 映射成三個 beta。

beta 一律依照

beta_{i,t,k} = b_k + a_k^T Z_{i,t}

其中 Z_{i,t} = [Z_size, Z_btp]。
observable firm characteristics 的 exp() 映射保留給輸出與展示，
不參與 beta 計算。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from toy_ff_generator.characteristics import LATENT_STATE_COLUMNS, LATENT_STATE_DIM


def _coerce_loading_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (LATENT_STATE_DIM,):
        raise ValueError(
            f"{name} must have shape ({LATENT_STATE_DIM},) for "
            f"{LATENT_STATE_COLUMNS}. Received {vector.shape}."
        )
    return vector


def generate_exposures(
    latent_state_df: pd.DataFrame,
    a_mkt: Sequence[float],
    b_mkt: float,
    a_smb: Sequence[float],
    b_smb: float,
    a_hml: Sequence[float],
    b_hml: float,
) -> pd.DataFrame:
    """依照 latent `Z_size` / `Z_btp` state 生成三個 beta。"""

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
    a_mkt_vector = _coerce_loading_vector(a_mkt, "a_mkt")
    a_smb_vector = _coerce_loading_vector(a_smb, "a_smb")
    a_hml_vector = _coerce_loading_vector(a_hml, "a_hml")

    beta_df = latent_state_df[["stock_id", "t"]].copy()
    beta_df["beta_mkt"] = latent_state_matrix @ a_mkt_vector + float(b_mkt)
    beta_df["beta_smb"] = latent_state_matrix @ a_smb_vector + float(b_smb)
    beta_df["beta_hml"] = latent_state_matrix @ a_hml_vector + float(b_hml)
    return beta_df

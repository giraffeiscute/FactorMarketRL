"""
這個模組負責把三維 characteristic vector 映射成三個 beta。

新版公式為：

beta_{i,t,k} = b_k + a_k^T C_{i,t}

其中 C_{i,t} = [C1, C2, C3]^T，a_k 為長度 3 向量。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def generate_exposures(
    characteristic_df: pd.DataFrame,
    a_mkt: Sequence[float],
    b_mkt: float,
    a_smb: Sequence[float],
    b_smb: float,
    a_hml: Sequence[float],
    b_hml: float,
) -> pd.DataFrame:
    """依照 `b + a^T C` 規則生成三個 beta。"""

    characteristic_matrix = characteristic_df[["C1", "C2", "C3"]].to_numpy(dtype=float)
    a_mkt_vector = np.asarray(a_mkt, dtype=float)
    a_smb_vector = np.asarray(a_smb, dtype=float)
    a_hml_vector = np.asarray(a_hml, dtype=float)

    beta_df = characteristic_df[["stock_id", "t"]].copy()
    beta_df["beta_mkt"] = characteristic_matrix @ a_mkt_vector + float(b_mkt)
    beta_df["beta_smb"] = characteristic_matrix @ a_smb_vector + float(b_smb)
    beta_df["beta_hml"] = characteristic_matrix @ a_hml_vector + float(b_hml)
    return beta_df

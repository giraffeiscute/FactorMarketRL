"""
這個模組放整個專案共用的工具函式與輸出存檔函式。

閱讀順序刻意整理成：
1. 一般工具
2. 矩陣轉換工具
3. 輸出存檔工具
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def set_random_seed(seed: int) -> np.random.Generator:
    """建立可重現的 NumPy 亂數產生器。"""

    return np.random.default_rng(seed)


def make_stock_ids(n: int) -> list[str]:
    """建立標準化股票代號，例如 `stock_000`。"""

    return [f"stock_{idx:03d}" for idx in range(n)]


def make_time_columns(t_count: int) -> list[str]:
    """建立標準化時間欄位名稱，例如 `t_0`、`t_1`。"""

    return [f"t_{idx}" for idx in range(t_count)]


def ensure_output_dir(output_dir: str | Path) -> Path:
    """確認輸出資料夾存在，若不存在就建立。"""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def pivot_to_wide_matrix(
    df: pd.DataFrame,
    value_col: str,
    time_columns: Sequence[str],
    index_col: str = "stock_id",
    column_col: str = "t",
) -> pd.DataFrame:
    """把 long panel 轉成 `股票 x 時間` 的矩陣格式。"""

    wide_df = df.pivot(index=index_col, columns=column_col, values=value_col)
    wide_df = wide_df.reindex(columns=list(time_columns))
    wide_df.index.name = index_col
    return wide_df.sort_index()


def save_outputs(
    panel_long_df: pd.DataFrame,
    output_dir: str | Path,
    time_columns: Sequence[str],
    metadata: Mapping[str, Any],
) -> dict[str, Path]:
    """把所有要求的輸出檔寫入指定資料夾。"""

    output_path = ensure_output_dir(output_dir)

    returns_wide = pivot_to_wide_matrix(
        df=panel_long_df,
        value_col="return",
        time_columns=time_columns,
    )
    prices_wide = pivot_to_wide_matrix(
        df=panel_long_df,
        value_col="price",
        time_columns=time_columns,
    )

    returns_path = output_path / "returns.csv"
    prices_path = output_path / "prices.csv"
    panel_path = output_path / "panel_long.csv"
    metadata_path = output_path / "metadata.json"

    _write_csv_atomically(returns_wide, returns_path, index=True)
    _write_csv_atomically(prices_wide, prices_path, index=True)
    _write_csv_atomically(panel_long_df, panel_path, index=False)
    _write_text_atomically(
        metadata_path,
        json.dumps(dict(metadata), indent=2),
    )

    return {
        "returns": returns_path,
        "prices": prices_path,
        "panel_long": panel_path,
        "metadata": metadata_path,
    }


def _write_csv_atomically(df: pd.DataFrame, path: Path, index: bool) -> None:
    """先寫暫存檔，再替換正式檔案，降低 Windows 覆寫問題。"""

    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        df.to_csv(temp_path, index=index)
        os.replace(temp_path, path)
    except PermissionError as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise PermissionError(
            f"Cannot write to {path}. "
            "If the file is open in another program, close it and retry."
        ) from exc


def _write_text_atomically(path: Path, content: str) -> None:
    """先寫暫存文字檔，再替換正式檔案。"""

    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        os.replace(temp_path, path)
    except PermissionError as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise PermissionError(
            f"Cannot write to {path}. "
            "If the file is open in another program, close it and retry."
        ) from exc

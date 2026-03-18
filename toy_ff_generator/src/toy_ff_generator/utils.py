from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from toy_ff_generator.characteristics import FIRM_CHARACTERISTIC_COLUMNS


def set_random_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_stock_ids(n: int) -> list[str]:
    return [f"stock_{idx:03d}" for idx in range(n)]


def make_time_columns(t_count: int) -> list[str]:
    return [f"t_{idx}" for idx in range(t_count)]


def ensure_output_dir(output_dir: str | Path) -> Path:
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
    wide_df = df.pivot(index=index_col, columns=column_col, values=value_col)
    wide_df = wide_df.reindex(columns=list(time_columns))
    wide_df.index.name = index_col
    return wide_df.sort_index()


def build_firm_characteristics_excel_view(
    firm_characteristics_df: pd.DataFrame,
) -> pd.DataFrame:
    excel_view = (
        firm_characteristics_df.set_index(["stock_id", "t"])[FIRM_CHARACTERISTIC_COLUMNS]
        .T
        .sort_index(axis=1)
    )
    excel_view.index.name = "firm_characteristic"
    return excel_view


def save_outputs(
    panel_long_df: pd.DataFrame,
    output_dir: str | Path,
    panel_filename: str,
    price_filename: str,
    return_filename: str,
    metadata_filename: str,
    time_columns: Sequence[str],
    metadata: Mapping[str, Any],
) -> dict[str, Path | None]:
    output_path = ensure_output_dir(output_dir)

    prices_wide = pivot_to_wide_matrix(
        df=panel_long_df,
        value_col="price",
        time_columns=time_columns,
    )
    returns_wide = pivot_to_wide_matrix(
        df=panel_long_df,
        value_col="return",
        time_columns=time_columns,
    )

    prices_path = output_path / price_filename
    returns_path = output_path / return_filename
    panel_path = output_path / panel_filename
    metadata_path = output_path / metadata_filename

    _write_csv_atomically(prices_wide, prices_path, index=True)
    _write_csv_atomically(returns_wide, returns_path, index=True)
    _write_csv_atomically(panel_long_df, panel_path, index=False)
    _write_text_atomically(
        metadata_path,
        json.dumps(dict(metadata), indent=2),
    )

    return {
        "prices": prices_path,
        "returns": returns_path,
        "panel_long": panel_path,
        "metadata": metadata_path,
        "excel_workbook": None,
    }


def _write_csv_atomically(df: pd.DataFrame, path: Path, index: bool) -> None:
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

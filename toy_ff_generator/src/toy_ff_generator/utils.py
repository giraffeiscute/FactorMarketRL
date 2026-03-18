from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

from toy_ff_generator.characteristics import FIRM_CHARACTERISTIC_COLUMNS

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PANEL_LONG_PERCENT_COLUMNS = (
    "alpha",
    "epsilon_variance",
    "MKT",
    "SMB",
    "HML",
    "epsilon",
    "raw_return",
    "return",
)


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


def _format_panel_long_for_csv(panel_long_df: pd.DataFrame) -> pd.DataFrame:
    formatted_df = panel_long_df.copy()
    for column_name in PANEL_LONG_PERCENT_COLUMNS:
        if column_name in formatted_df.columns:
            formatted_df[column_name] = formatted_df[column_name].map(
                lambda value: f"{float(value) * 100:.3f}%"
            )
    return formatted_df


def build_market_index_df(
    panel_long_df: pd.DataFrame,
    time_columns: Sequence[str],
) -> pd.DataFrame:
    market_index_df = (
        panel_long_df.groupby("t", as_index=False)
        .agg(
            market_index=("price", "mean"),
            MKT=("MKT", "first"),
            SMB=("SMB", "first"),
            HML=("HML", "first"),
        )
        .set_index("t")
        .reindex(list(time_columns))
        .reset_index()
    )
    return market_index_df


def _save_market_index_png(
    market_index_df: pd.DataFrame,
    path: Path,
    title: str,
) -> None:
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    figure, (market_axis, factor_axis) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 6.5),
        sharex=True,
    )
    x_values = market_index_df["t"].map(lambda value: int(str(value).split("_")[-1]))
    max_tick_count = 12
    tick_step = max(1, int(np.ceil(len(x_values) / max_tick_count)))
    tick_values = x_values.iloc[::tick_step].tolist()
    if tick_values and tick_values[-1] != int(x_values.iloc[-1]):
        tick_values.append(int(x_values.iloc[-1]))
    try:
        market_axis.plot(
            x_values,
            market_index_df["market_index"],
            linewidth=1.5,
            label="market_index",
        )
        market_axis.set_ylabel("market_index")
        market_axis.set_title(title)
        market_axis.grid(True, alpha=0.3)
        market_axis.legend(loc="best")

        for factor_name in ("MKT", "SMB", "HML"):
            factor_axis.plot(
                x_values,
                market_index_df[factor_name],
                linewidth=1.2,
                label=factor_name,
            )
        factor_axis.set_xlabel("t")
        factor_axis.set_ylabel("factor")
        factor_axis.set_xticks(tick_values)
        factor_axis.grid(True, alpha=0.3)
        factor_axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(temp_path, dpi=150)
        os.replace(temp_path, path)
    except PermissionError as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise PermissionError(
            f"Cannot write to {path}. "
            "If the file is open in another program, close it and retry."
        ) from exc
    finally:
        plt.close(figure)


def save_outputs(
    panel_long_df: pd.DataFrame,
    output_dir: str | Path,
    panel_filename: str,
    price_filename: str,
    return_filename: str,
    market_index_csv_filename: str,
    market_index_png_filename: str,
    market_index_plot_title: str,
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
    market_index_csv_path = output_path / market_index_csv_filename
    market_index_png_path = output_path / market_index_png_filename
    metadata_path = output_path / metadata_filename
    panel_long_output_df = _format_panel_long_for_csv(panel_long_df)
    market_index_df = build_market_index_df(
        panel_long_df=panel_long_df,
        time_columns=time_columns,
    )

    _write_csv_atomically(prices_wide, prices_path, index=True)
    _write_csv_atomically(returns_wide, returns_path, index=True)
    _write_csv_atomically(panel_long_output_df, panel_path, index=False)
    _write_csv_atomically(market_index_df, market_index_csv_path, index=False)
    _save_market_index_png(
        market_index_df=market_index_df,
        path=market_index_png_path,
        title=market_index_plot_title,
    )
    _write_text_atomically(
        metadata_path,
        json.dumps(dict(metadata), indent=2),
    )

    return {
        "prices": prices_path,
        "returns": returns_path,
        "panel_long": panel_path,
        "market_index_csv": market_index_csv_path,
        "market_index_png": market_index_png_path,
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

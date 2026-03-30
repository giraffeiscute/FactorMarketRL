from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from toy_ff_generator.characteristics import FIRM_CHARACTERISTIC_COLUMNS

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "toy_ff_generator_matplotlib"))

_PYPLOT_MODULE = None


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


def _to_time_index(raw_value: object) -> int:
    if isinstance(raw_value, str):
        normalized = raw_value.strip()
        if normalized.startswith("t_"):
            return int(normalized.split("_", 1)[1])
        return int(normalized)
    return int(raw_value)


def prepare_panel_long_for_parquet(panel_long_df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = panel_long_df.copy()
    prepared_df["t"] = prepared_df["t"].map(_to_time_index).astype("int64")
    return prepared_df


def build_market_index_df(
    panel_long_df: pd.DataFrame,
    time_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    if time_indices is None:
        time_index_values = (
            pd.Index(pd.to_numeric(panel_long_df["t"], errors="raise"))
            .sort_values()
            .unique()
            .tolist()
        )
    else:
        time_index_values = [int(value) for value in time_indices]
    market_index_df = (
        panel_long_df.groupby("t", as_index=False)
        .agg(
            market_index=("price", "mean"),
            price_std=("price", lambda values: values.std(ddof=0)),
            price_min=("price", "min"),
            price_max=("price", "max"),
            MKT=("MKT", "first"),
            SMB=("SMB", "first"),
            HML=("HML", "first"),
        )
        .set_index("t")
        .reindex(time_index_values)
        .reset_index()
    )
    market_index_df["t"] = pd.to_numeric(market_index_df["t"], errors="raise").astype("int64")
    return market_index_df


def _save_market_index_png(
    market_index_df: pd.DataFrame,
    path: Path,
    title: str,
) -> None:
    plt = _get_pyplot()
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    figure, (market_axis, factor_axis) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 6.5),
        sharex=True,
    )
    x_values = pd.to_numeric(market_index_df["t"], errors="raise").astype(int)
    market_values = market_index_df["market_index"].to_numpy(dtype=float)
    price_std_values = market_index_df["price_std"].to_numpy(dtype=float)
    price_min_values = market_index_df["price_min"].to_numpy(dtype=float)
    price_max_values = market_index_df["price_max"].to_numpy(dtype=float)
    avg_price_std = float(
        market_index_df["price_std"].dropna().astype(float).mean()
    )
    std_lower_band = market_values - price_std_values
    std_upper_band = market_values + price_std_values
    max_tick_count = 12
    tick_step = max(1, int(np.ceil(len(x_values) / max_tick_count)))
    tick_values = x_values.iloc[::tick_step].tolist()
    if tick_values and tick_values[-1] != int(x_values.iloc[-1]):
        tick_values.append(int(x_values.iloc[-1]))
    try:
        market_line_color = "tab:blue"
        market_axis.fill_between(
            x_values,
            price_min_values,
            price_max_values,
            color=market_line_color,
            alpha=0.08,
            label="min/max band",
        )
        market_axis.fill_between(
            x_values,
            std_lower_band,
            std_upper_band,
            color=market_line_color,
            alpha=0.2,
            label="std band",
        )
        market_axis.plot(
            x_values,
            market_values,
            linewidth=1.8,
            color=market_line_color,
            label="market_index",
        )
        market_axis.text(
            0.02,
            0.98,
            f"avg_std = {avg_price_std:.4f}",
            transform=market_axis.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
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


def _get_pyplot():
    global _PYPLOT_MODULE
    if _PYPLOT_MODULE is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _PYPLOT_MODULE = plt
    return _PYPLOT_MODULE


def _write_parquet_atomically(df: pd.DataFrame, path: Path, index: bool) -> None:
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        df.to_parquet(temp_path, index=index)
        os.replace(temp_path, path)
    except PermissionError as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise PermissionError(
            f"Cannot write to {path}. "
            "If the file is open in another program, close it and retry."
        ) from exc


def save_outputs(
    panel_long_df: pd.DataFrame,
    output_dir: str | Path,
    panel_filename: str,
    market_index_csv_filename: str,
    market_index_png_filename: str,
    market_index_plot_title: str,
    time_columns: Sequence[str],
) -> dict[str, Path | None]:
    output_path = ensure_output_dir(output_dir)
    panel_path = output_path / panel_filename
    market_index_csv_path = output_path / market_index_csv_filename
    market_index_png_path = output_path / market_index_png_filename
    del time_columns

    panel_long_output_df = prepare_panel_long_for_parquet(panel_long_df)
    _write_parquet_atomically(panel_long_output_df, panel_path, index=False)
    rebuild_market_index_artifacts_from_panel_frame(
        panel_long_df=panel_long_output_df,
        market_index_csv_path=market_index_csv_path,
        market_index_png_path=market_index_png_path,
        title=market_index_plot_title,
    )

    return {
        "prices": None,
        "returns": None,
        "panel_long": panel_path,
        "market_index_csv": market_index_csv_path,
        "market_index_png": market_index_png_path,
        "metadata": None,
        "excel_workbook": None,
    }


def rebuild_market_index_artifacts_from_panel_frame(
    *,
    panel_long_df: pd.DataFrame,
    market_index_csv_path: Path,
    market_index_png_path: Path,
    title: str,
) -> pd.DataFrame:
    numeric_panel = prepare_panel_long_for_parquet(panel_long_df)
    market_index_df = build_market_index_df(panel_long_df=numeric_panel)
    _write_csv_atomically(market_index_df, market_index_csv_path, index=False)
    _save_market_index_png(
        market_index_df=market_index_df,
        path=market_index_png_path,
        title=title,
    )
    return market_index_df


def rebuild_market_index_artifacts_from_panel_path(
    *,
    panel_path: str | Path,
    market_index_csv_path: str | Path,
    market_index_png_path: str | Path,
    title: str,
) -> pd.DataFrame:
    panel_path = Path(panel_path)
    panel_long_df = pd.read_parquet(panel_path)
    return rebuild_market_index_artifacts_from_panel_frame(
        panel_long_df=panel_long_df,
        market_index_csv_path=Path(market_index_csv_path),
        market_index_png_path=Path(market_index_png_path),
        title=title,
    )


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

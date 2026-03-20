"""Dataset loading and validation for portfolio_attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DataConfig

REQUIRED_COLUMNS = [
    "stock_id",
    "t",
    "characteristic_1",
    "characteristic_2",
    "characteristic_3",
    "MKT",
    "SMB",
    "HML",
    "price",
]
STOCK_FEATURE_COLUMNS = [
    "characteristic_1",
    "characteristic_2",
    "characteristic_3",
    "price",
]
MARKET_FEATURE_COLUMNS = ["MKT", "SMB", "HML"]
NUMERIC_COLUMNS = STOCK_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS


def parse_panel_dimensions(file_name: str) -> tuple[int, int]:
    """Parse N and T from `{prefix_}N_T_panel_long.csv`."""

    match = re.search(r"(?:^|_)(?P<n>\d+)_(?P<t>\d+)_panel_long\.csv$", file_name)
    if not match:
        raise ValueError(f"Could not parse N/T from file name: {file_name}")
    return int(match.group("n")), int(match.group("t"))


def _parse_time_label(raw_value: Any) -> int:
    if isinstance(raw_value, str):
        match = re.fullmatch(r"t_(\d+)", raw_value)
        if not match:
            raise ValueError(f"Unsupported time label: {raw_value}")
        return int(match.group(1))
    return int(raw_value)


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="raise")

    normalized = series.astype(str).str.strip()
    percent_mask = normalized.str.endswith("%")
    if percent_mask.any():
        normalized = normalized.str.replace("%", "", regex=False)
        numeric = pd.to_numeric(normalized, errors="raise")
        return numeric / 100.0
    return pd.to_numeric(normalized, errors="raise")


class Standardizer:
    """Simple ndarray standardizer fit only on training rows."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "Standardizer":
        if values.size == 0:
            raise ValueError("Cannot fit a scaler on empty values.")
        self.mean = values.mean(axis=0)
        std = values.std(axis=0)
        self.std = np.where(std < 1e-6, 1.0, std)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer must be fit before transform.")
        return (values - self.mean) / self.std


@dataclass
class PanelMetadata:
    """Dataset summary with honest sample accounting."""

    source_path: str
    parsed_n: int
    parsed_t: int
    csv_unique_stocks: int
    csv_unique_times: int
    train_days: int
    test_days: int
    lookback: int
    analysis_entry_day: int
    analysis_exit_day: int
    legal_train_windows: int
    legal_test_windows: int
    available_analysis_windows: int
    analysis_only: bool
    selected_num_stocks: int
    effective_time_steps: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PortfolioWindowDataset(Dataset):
    """Rolling-window dataset used by train.py for epoch-based training."""

    def __init__(self, panel_dataset: "PortfolioPanelDataset", start_indices: list[int]) -> None:
        self.panel_dataset = panel_dataset
        self.start_indices = start_indices

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start_index = self.start_indices[index]
        window = self.panel_dataset.get_window(start_index)
        return {
            key: torch.from_numpy(value)
            for key, value in window.items()
        }


class PortfolioPanelDataset:
    """Loads the long-format panel and builds a fixed-stock dataset.

    Time position uses `x_t = w_t + p_t`.
    Stock identity position is handled later in the model as `x_{s,t} = [z_{s,t}; e_s]`.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.csv_path = Path(config.csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")

        self.parsed_n, self.parsed_t = parse_panel_dimensions(self.csv_path.name)
        self.lookback = config.lookback
        self._load()

    def _validate_num_stocks(self) -> None:
        if self.config.num_stocks is None:
            return
        if self.config.num_stocks <= 0:
            raise ValueError("DataConfig.num_stocks must be positive when provided.")
        actual_num_stocks = len(self.stock_ids)
        if self.config.num_stocks != actual_num_stocks:
            raise ValueError(
                f"DataConfig.num_stocks={self.config.num_stocks} does not match the dataset stock count "
                f"{actual_num_stocks} inferred from {self.csv_path.name}."
            )

    def _load(self) -> None:
        header = pd.read_csv(self.csv_path, nrows=0).columns.tolist()
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in header]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        self.ignored_extra_columns = [column for column in header if column not in REQUIRED_COLUMNS]
        self.loaded_stock_feature_columns = list(STOCK_FEATURE_COLUMNS)
        self.loaded_market_feature_columns = list(MARKET_FEATURE_COLUMNS)

        frame = pd.read_csv(self.csv_path, usecols=REQUIRED_COLUMNS)
        for column in NUMERIC_COLUMNS:
            frame[column] = _coerce_numeric_series(frame[column])
        frame["time_index"] = frame["t"].map(_parse_time_label)
        frame = frame.sort_values(["stock_id", "time_index"], kind="mergesort").reset_index(drop=True)

        if frame.duplicated(["stock_id", "time_index"]).any():
            raise ValueError("Panel contains duplicated (stock_id, t) rows.")

        self.stock_ids = sorted(frame["stock_id"].unique().tolist())
        self.time_index = sorted(frame["time_index"].unique().tolist())

        if len(self.stock_ids) != self.parsed_n:
            raise ValueError(
                f"Parsed N={self.parsed_n} from file name but CSV contains {len(self.stock_ids)} stocks."
            )
        if len(self.time_index) != self.parsed_t:
            raise ValueError(
                f"Parsed T={self.parsed_t} from file name but CSV contains {len(self.time_index)} time points."
            )
        self._validate_num_stocks()

        full_index = pd.MultiIndex.from_product(
            [self.stock_ids, self.time_index],
            names=["stock_id", "time_index"],
        )
        indexed = frame.set_index(["stock_id", "time_index"]).sort_index()
        if len(indexed) != len(full_index):
            raise ValueError("Panel is incomplete: row count does not match N * T.")
        reindexed = indexed.reindex(full_index)
        if reindexed.isna().any().any():
            raise ValueError("Panel is incomplete: at least one (stock_id, t) combination is missing.")

        ff3_consistency = frame.groupby("time_index")[MARKET_FEATURE_COLUMNS].nunique(dropna=False)
        if (ff3_consistency > 1).any().any():
            raise ValueError("FF3 factors are not identical across all stocks within the same day.")

        stock_arrays = []
        for column in STOCK_FEATURE_COLUMNS:
            pivot = (
                reindexed[column]
                .unstack("time_index")
                .reindex(index=self.stock_ids, columns=self.time_index)
                .to_numpy(dtype=np.float32)
            )
            stock_arrays.append(pivot)
        self.stock_features_raw = np.stack(stock_arrays, axis=-1)
        self.price_array = self.stock_features_raw[..., -1].copy()

        self.market_features_raw = (
            frame.groupby("time_index")[MARKET_FEATURE_COLUMNS]
            .first()
            .reindex(self.time_index)
            .to_numpy(dtype=np.float32)
        )

        self.train_days = int(self.parsed_t * self.config.train_ratio)
        self.test_days = self.parsed_t - self.train_days
        self.analysis_entry_day = self.config.resolved_entry_day()
        default_exit_day = self.parsed_t - 1
        self.analysis_exit_day = self.config.analysis_exit_day or default_exit_day
        if self.analysis_exit_day <= self.analysis_entry_day:
            raise ValueError("analysis_exit_day must be greater than analysis_entry_day.")
        if self.analysis_exit_day > self.parsed_t:
            raise ValueError("analysis_exit_day exceeds the available time range.")

        self.entry_index = self.analysis_entry_day - 1
        self.exit_index = self.analysis_exit_day - 1
        self.holding_period = self.exit_index - self.entry_index

        self.stock_scaler = Standardizer().fit(
            self.stock_features_raw[:, : self.train_days, :].reshape(-1, len(STOCK_FEATURE_COLUMNS))
        )
        self.market_scaler = Standardizer().fit(self.market_features_raw[: self.train_days])
        self.stock_features_scaled = self.stock_scaler.transform(self.stock_features_raw)
        self.market_features_scaled = self.market_scaler.transform(self.market_features_raw)

        self.selected_stock_ids = list(self.stock_ids)
        selected_n = len(self.selected_stock_ids)

        self.legal_train_windows = self._count_legal_windows(self.train_days)
        self.legal_test_windows = self._count_legal_windows(self.test_days)
        self.analysis_window_start_index = self.entry_index - self.lookback
        self.available_analysis_windows = int(
            0 <= self.analysis_window_start_index
            and self.entry_index < self.exit_index
            and self.exit_index < self.parsed_t
        )

        self.metadata = PanelMetadata(
            source_path=str(self.csv_path),
            parsed_n=self.parsed_n,
            parsed_t=self.parsed_t,
            csv_unique_stocks=len(self.stock_ids),
            csv_unique_times=len(self.time_index),
            train_days=self.train_days,
            test_days=self.test_days,
            lookback=self.lookback,
            analysis_entry_day=self.analysis_entry_day,
            analysis_exit_day=self.analysis_exit_day,
            legal_train_windows=self.legal_train_windows,
            legal_test_windows=self.legal_test_windows,
            available_analysis_windows=self.available_analysis_windows,
            analysis_only=True,
            selected_num_stocks=selected_n,
            effective_time_steps=self.parsed_t,
        )

        if self.legal_train_windows == 0 and self.legal_test_windows == 0:
            warnings.warn(
                "This dataset yields 0 legal train windows and 0 legal test windows under the fixed sample definition. "
                "Only the single cross-boundary analysis window is available.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _count_legal_windows(self, split_length: int) -> int:
        return max(0, split_length - self.lookback - self.holding_period)

    @property
    def total_window_count(self) -> int:
        return max(0, self.parsed_t - self.lookback - self.holding_period)

    @property
    def num_stocks(self) -> int:
        return len(self.selected_stock_ids)

    @property
    def num_times(self) -> int:
        return self.parsed_t

    def get_window(self, start_index: int) -> dict[str, np.ndarray]:
        if start_index < 0 or start_index >= self.total_window_count:
            raise IndexError(f"Window start_index={start_index} is out of range for total_window_count={self.total_window_count}.")

        lookback_stop = start_index + self.lookback
        entry_index = lookback_stop
        exit_index = lookback_stop + self.holding_period

        x_stock = self.stock_features_scaled[:, start_index:lookback_stop, :]
        x_market = self.market_features_scaled[start_index:lookback_stop]
        entry_prices = self.price_array[:, entry_index]
        exit_prices = self.price_array[:, exit_index]
        r_stock = (exit_prices / entry_prices) - 1.0
        stock_indices = np.arange(self.num_stocks, dtype=np.int64)

        return {
            "x_stock": x_stock.astype(np.float32),
            "x_market": x_market.astype(np.float32),
            "r_stock": r_stock.astype(np.float32),
            "stock_indices": stock_indices,
        }

    def get_train_val_window_indices(self) -> tuple[list[int], list[int]]:
        if self.total_window_count < 2:
            return [], []

        train_count = max(1, int(self.total_window_count * self.config.train_ratio))
        if train_count >= self.total_window_count:
            train_count = self.total_window_count - 1

        train_indices = list(range(train_count))
        val_indices = list(range(train_count, self.total_window_count))
        return train_indices, val_indices

    def build_train_val_datasets(self) -> tuple[PortfolioWindowDataset, PortfolioWindowDataset]:
        train_indices, val_indices = self.get_train_val_window_indices()
        return PortfolioWindowDataset(self, train_indices), PortfolioWindowDataset(self, val_indices)

    def get_analysis_window(self) -> dict[str, np.ndarray]:
        if self.available_analysis_windows != 1:
            raise RuntimeError(
                "The configured analysis window is unavailable. "
                f"entry_day={self.analysis_entry_day}, exit_day={self.analysis_exit_day}, "
                f"lookback={self.lookback}, total_days={self.parsed_t}."
            )

        window = self.get_window(start_index=self.analysis_window_start_index)
        return {
            "x_stock": window["x_stock"][np.newaxis, ...],
            "x_market": window["x_market"][np.newaxis, ...],
            "r_stock": window["r_stock"][np.newaxis, ...],
            "stock_indices": window["stock_indices"][np.newaxis, ...],
        }

    def get_analysis_batch(self, device: torch.device | None = None) -> dict[str, torch.Tensor]:
        batch = self.get_analysis_window()
        tensor_batch = {
            key: torch.as_tensor(value, device=device)
            for key, value in batch.items()
        }
        tensor_batch["metadata"] = self.metadata.as_dict()
        return tensor_batch

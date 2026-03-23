"""Dataset loading and validation for portfolio_attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
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
    """Dataset summary for the single fixed train/test sample rule."""

    source_path: str
    parsed_n: int
    parsed_t: int
    csv_unique_stocks: int
    csv_unique_times: int
    total_num_days: int
    train_days: int
    test_days: int
    train_split_length: int
    test_split_length: int
    analysis_horizon_days: int
    train_horizon_start_index: int
    train_horizon_end_index: int
    test_horizon_start_index: int
    test_horizon_end_index: int
    dynamic_train_lookback_length: int
    dynamic_test_lookback_length: int
    model_lookback: int
    legal_train_windows: int
    legal_test_windows: int
    train_window_count: int
    test_window_count: int
    available_analysis_windows: int
    analysis_only: bool
    selected_num_stocks: int
    effective_time_steps: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindowSpec:
    """Defines one cross-sectional sample window inside the panel."""

    split_name: str
    start_index: int
    lookback_length: int
    horizon_start_index: int
    horizon_end_index: int


class PortfolioWindowDataset(Dataset):
    """Single-window dataset used by train.py for epoch-based training."""

    def __init__(
        self,
        panel_dataset: "PortfolioPanelDataset",
        window_specs: list[WindowSpec],
    ) -> None:
        self.panel_dataset = panel_dataset
        self.window_specs = window_specs

    def __len__(self) -> int:
        return len(self.window_specs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.panel_dataset.get_window(self.window_specs[index])
        return {
            key: torch.from_numpy(value)
            for key, value in window.items()
        }


class PortfolioPanelDataset:
    """Loads the long-format panel and builds the fixed train/test samples.

    The project now uses one and only one train sample plus one and only one
    test sample. Each sample keeps the full cross-sectional stock universe and
    uses all history before its horizon start as lookback.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.csv_path = Path(config.csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")

        self.parsed_n, self.parsed_t = parse_panel_dimensions(self.csv_path.name)
        self.model_lookback = 0
        self._load()

    def _resolve_selected_stock_indices(self) -> np.ndarray:
        actual_num_stocks = len(self.stock_ids)
        requested_num_stocks = self.config.num_stocks
        if requested_num_stocks is None:
            return np.arange(actual_num_stocks, dtype=np.int64)
        if requested_num_stocks <= 0:
            raise ValueError("DataConfig.num_stocks must be positive when provided.")
        if requested_num_stocks > actual_num_stocks:
            raise ValueError(
                f"Requested fixed num_stocks={requested_num_stocks}, "
                f"but dataset only provides {actual_num_stocks} stocks."
            )
        return np.arange(requested_num_stocks, dtype=np.int64)

    def _build_window_spec(
        self,
        *,
        split_name: str,
        horizon_start_index: int,
        horizon_end_index: int,
        lookback_length: int,
    ) -> WindowSpec:
        if lookback_length <= 0:
            raise ValueError(
                f"dynamic_{split_name}_lookback_length must be positive, received {lookback_length}."
            )
        return WindowSpec(
            split_name=split_name,
            start_index=0,
            lookback_length=lookback_length,
            horizon_start_index=horizon_start_index,
            horizon_end_index=horizon_end_index,
        )

    def _initialize_single_sample_windows(self) -> None:
        analysis_horizon_days = int(self.config.analysis_horizon_days)
        if analysis_horizon_days <= 0:
            raise ValueError("analysis_horizon_days must be positive.")

        train_split_length = self.train_days
        test_split_length = self.test_days
        if train_split_length <= analysis_horizon_days:
            raise ValueError(
                f"train_split_length={train_split_length} must be greater than analysis_horizon_days="
                f"{analysis_horizon_days} so the train split can contain the full horizon and leave a valid lookback."
            )
        if test_split_length <= analysis_horizon_days:
            raise ValueError(
                f"test_split_length={test_split_length} must be greater than analysis_horizon_days="
                f"{analysis_horizon_days} so the test split can contain the full horizon and leave a valid lookback."
            )

        self.analysis_horizon_days = analysis_horizon_days
        self.train_horizon_start_index = train_split_length - analysis_horizon_days
        self.train_horizon_end_index = train_split_length - 1
        self.test_horizon_start_index = self.parsed_t - analysis_horizon_days
        self.test_horizon_end_index = self.parsed_t - 1
        self.dynamic_train_lookback_length = self.train_horizon_start_index
        self.dynamic_test_lookback_length = self.test_horizon_start_index

        if self.dynamic_train_lookback_length <= 0:
            raise ValueError(
                "dynamic_train_lookback_length must be positive under the fixed single-sample rule. "
                f"Received {self.dynamic_train_lookback_length}."
            )
        if self.dynamic_test_lookback_length <= 0:
            raise ValueError(
                "dynamic_test_lookback_length must be positive under the fixed single-sample rule. "
                f"Received {self.dynamic_test_lookback_length}."
            )
        if self.train_horizon_end_index >= self.train_days:
            raise ValueError("Train horizon must lie fully inside the train split.")
        if self.test_horizon_start_index < self.train_days:
            raise ValueError("Test horizon must lie fully inside the test split.")

        train_spec = self._build_window_spec(
            split_name="train",
            horizon_start_index=self.train_horizon_start_index,
            horizon_end_index=self.train_horizon_end_index,
            lookback_length=self.dynamic_train_lookback_length,
        )
        test_spec = self._build_window_spec(
            split_name="test",
            horizon_start_index=self.test_horizon_start_index,
            horizon_end_index=self.test_horizon_end_index,
            lookback_length=self.dynamic_test_lookback_length,
        )

        self.train_window_specs = [train_spec]
        self.test_window_specs = [test_spec]
        self.analysis_window_spec = test_spec
        self.model_lookback = max(
            self.dynamic_train_lookback_length,
            self.dynamic_test_lookback_length,
        )
        self.legal_train_windows = 1
        self.legal_test_windows = 1
        self.available_analysis_windows = 1

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
        full_stock_features_raw = np.stack(stock_arrays, axis=-1)

        self.market_features_raw = (
            frame.groupby("time_index")[MARKET_FEATURE_COLUMNS]
            .first()
            .reindex(self.time_index)
            .to_numpy(dtype=np.float32)
        )

        selected_stock_indices = self._resolve_selected_stock_indices()
        self.selected_stock_ids = [self.stock_ids[index] for index in selected_stock_indices.tolist()]
        self.stock_features_raw = full_stock_features_raw[selected_stock_indices]
        self.price_array = self.stock_features_raw[..., -1].copy()

        if not 0.0 < self.config.train_ratio < 1.0:
            raise ValueError("DataConfig.train_ratio must be between 0 and 1.")
        self.train_days = int(self.parsed_t * self.config.train_ratio)
        self.test_days = self.parsed_t - self.train_days
        if self.train_days <= 0 or self.test_days <= 0:
            raise ValueError(
                "train_ratio must produce non-empty train and test splits. "
                f"Received train_days={self.train_days}, test_days={self.test_days}."
            )

        self.stock_scaler = Standardizer().fit(
            self.stock_features_raw[:, : self.train_days, :].reshape(-1, len(STOCK_FEATURE_COLUMNS))
        )
        self.market_scaler = Standardizer().fit(self.market_features_raw[: self.train_days])
        self.stock_features_scaled = self.stock_scaler.transform(self.stock_features_raw)
        self.market_features_scaled = self.market_scaler.transform(self.market_features_raw)

        self._initialize_single_sample_windows()

        self.metadata = PanelMetadata(
            source_path=str(self.csv_path),
            parsed_n=self.parsed_n,
            parsed_t=self.parsed_t,
            csv_unique_stocks=len(self.stock_ids),
            csv_unique_times=len(self.time_index),
            total_num_days=self.parsed_t,
            train_days=self.train_days,
            test_days=self.test_days,
            train_split_length=self.train_days,
            test_split_length=self.test_days,
            analysis_horizon_days=self.analysis_horizon_days,
            train_horizon_start_index=self.train_horizon_start_index,
            train_horizon_end_index=self.train_horizon_end_index,
            test_horizon_start_index=self.test_horizon_start_index,
            test_horizon_end_index=self.test_horizon_end_index,
            dynamic_train_lookback_length=self.dynamic_train_lookback_length,
            dynamic_test_lookback_length=self.dynamic_test_lookback_length,
            model_lookback=self.model_lookback,
            legal_train_windows=self.legal_train_windows,
            legal_test_windows=self.legal_test_windows,
            train_window_count=len(self.train_window_specs),
            test_window_count=len(self.test_window_specs),
            available_analysis_windows=self.available_analysis_windows,
            analysis_only=True,
            selected_num_stocks=len(self.selected_stock_ids),
            effective_time_steps=self.parsed_t,
        )

    @property
    def num_stocks(self) -> int:
        return len(self.selected_stock_ids)

    @property
    def num_times(self) -> int:
        return self.parsed_t

    def get_window(self, window_spec: WindowSpec) -> dict[str, np.ndarray]:
        lookback_start_index = window_spec.start_index
        lookback_stop_index = lookback_start_index + window_spec.lookback_length
        horizon_start_index = window_spec.horizon_start_index
        horizon_end_index = window_spec.horizon_end_index

        if lookback_start_index != 0:
            raise ValueError("The fixed single-sample rule expects lookback_start_index=0.")
        if lookback_stop_index != horizon_start_index:
            raise ValueError(
                "Lookback must end exactly where the horizon starts under the fixed single-sample rule."
            )
        if not (0 <= horizon_start_index < horizon_end_index < self.parsed_t):
            raise IndexError(
                "Invalid horizon indices for the requested window: "
                f"start={horizon_start_index}, end={horizon_end_index}, total_num_days={self.parsed_t}."
            )

        x_stock = self.stock_features_scaled[:, lookback_start_index:lookback_stop_index, :]
        x_market = self.market_features_scaled[lookback_start_index:lookback_stop_index]
        entry_prices = self.price_array[:, horizon_start_index]
        exit_prices = self.price_array[:, horizon_end_index]
        r_stock = (exit_prices / entry_prices) - 1.0
        stock_indices = np.arange(self.num_stocks, dtype=np.int64)

        return {
            "x_stock": x_stock.astype(np.float32),
            "x_market": x_market.astype(np.float32),
            "r_stock": r_stock.astype(np.float32),
            "stock_indices": stock_indices,
        }

    def build_train_val_datasets(self) -> tuple[PortfolioWindowDataset, PortfolioWindowDataset]:
        return (
            PortfolioWindowDataset(self, list(self.train_window_specs)),
            PortfolioWindowDataset(self, list(self.test_window_specs)),
        )

    def get_analysis_window(self) -> dict[str, np.ndarray]:
        if self.available_analysis_windows != 1:
            raise RuntimeError(
                "The configured test analysis window is unavailable. "
                f"test_horizon_start_index={self.test_horizon_start_index}, "
                f"test_horizon_end_index={self.test_horizon_end_index}, "
                f"lookback_length={self.analysis_window_spec.lookback_length}, "
                f"total_num_days={self.parsed_t}."
            )

        window = self.get_window(self.analysis_window_spec)
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

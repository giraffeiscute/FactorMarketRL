"""Scenario-aware dataset loading and validation for portfolio_attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .config import DataConfig

BASE_REQUIRED_COLUMNS = [
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
OPTIONAL_RETURN_COLUMN = "return"
STOCK_FEATURE_COLUMNS = [
    "characteristic_1",
    "characteristic_2",
    "characteristic_3",
    "price",
]
MARKET_FEATURE_COLUMNS = ["MKT", "SMB", "HML"]
NUMERIC_COLUMNS = STOCK_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS + [OPTIONAL_RETURN_COLUMN]
LOADABLE_COLUMNS = BASE_REQUIRED_COLUMNS + [OPTIONAL_RETURN_COLUMN]
SCENARIO_FILE_PATTERN = re.compile(
    r"^(?P<state>.+?)_(?P<n>\d+)_(?P<t>\d+)_PL_(?P<scenario_index>\d+)\.parquet$"
)


def parse_panel_dimensions(file_name: str) -> tuple[int, int]:
    """Parse `(N, T)` from a scenario file name like `{state}_{N}_{T}_PL_{idx}.parquet`."""

    match = SCENARIO_FILE_PATTERN.fullmatch(file_name)
    if not match:
        raise ValueError(f"Could not parse N/T from file name: {file_name}")
    return int(match.group("n")), int(match.group("t"))


def parse_scenario_file_info(file_name: str) -> dict[str, int | str]:
    match = SCENARIO_FILE_PATTERN.fullmatch(file_name)
    if not match:
        raise ValueError(f"Unsupported scenario file name: {file_name}")
    return {
        "state": match.group("state"),
        "parsed_n": int(match.group("n")),
        "parsed_t": int(match.group("t")),
        "scenario_index": int(match.group("scenario_index")),
    }


def _parse_time_label(raw_value: Any) -> int:
    if isinstance(raw_value, str):
        match = re.fullmatch(r"t_(\d+)", raw_value)
        if not match:
            raise ValueError(f"Unsupported time label: {raw_value}")
        return int(match.group(1))
    return int(raw_value)


def _parse_time_series(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return pd.to_numeric(series, errors="raise").to_numpy(dtype=np.int64, copy=False)

    normalized = series.astype(str).str.strip()
    if not normalized.str.startswith("t_").all():
        invalid = normalized[~normalized.str.startswith("t_")].iloc[0]
        raise ValueError(f"Unsupported time label: {invalid}")
    return normalized.str.slice(2).astype(np.int64).to_numpy(copy=False)


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
    """Simple ndarray standardizer fit only on train-scenario train-segment rows."""

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

    def set_statistics(self, mean: np.ndarray, std: np.ndarray) -> "Standardizer":
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer must be fit before transform.")
        return (values - self.mean) / self.std


class RunningMoments:
    """Streaming moments helper used to avoid fitting scalers on validation/test rows."""

    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = feature_dim
        self.count = 0
        self.sum = np.zeros((feature_dim,), dtype=np.float64)
        self.sum_sq = np.zeros((feature_dim,), dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected values with shape [*, {self.feature_dim}], received {values.shape}."
            )
        self.count += int(values.shape[0])
        self.sum += values.sum(axis=0, dtype=np.float64)
        self.sum_sq += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count <= 0:
            raise ValueError("Cannot finalize moments without any observations.")
        mean = self.sum / float(self.count)
        variance = (self.sum_sq / float(self.count)) - np.square(mean)
        variance = np.maximum(variance, 1e-12)
        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


@dataclass
class ScenarioFileRecord:
    scenario_id: str
    source_path: str
    state: str
    scenario_index: int
    parsed_n: int
    parsed_t: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioSegmentRecord:
    """One full scenario segment.

    Tensor layout for a single record:
    - `x_stock`: [T_split, N, F_stock]
    - `x_market`: [T_split, F_market]
    - `r_stock`: [T_split, N]
    - `stock_indices`: [N]

    A DataLoader stacks these into:
    - `x_stock`: [S, T_split, N, F_stock]
    - `x_market`: [S, T_split, F_market]
    - `r_stock`: [S, T_split, N]
    - `stock_indices`: [S, N]

    `S` is the scenario batch dimension and must never be flattened together with
    the time dimension `T_split`.
    """

    scenario_id: str
    source_path: str
    split_name: str
    feature_time_indices: np.ndarray
    target_time_indices: np.ndarray
    x_stock: np.ndarray
    x_market: np.ndarray
    r_stock: np.ndarray
    stock_indices: np.ndarray


@dataclass
class ScenarioDatasetMetadata:
    scenario_dir: str
    scenario_glob: str
    state: str
    total_scenarios_found: int
    num_train_scenarios: int
    num_validation_scenarios: int
    num_test_scenarios: int
    train_scenarios: list[str]
    validation_scenarios: list[str]
    test_scenarios: list[str]
    total_num_days: int
    train_segment_start_index: int
    train_segment_end_index: int
    validation_segment_start_index: int
    validation_segment_end_index: int
    test_segment_start_index: int
    test_segment_end_index: int
    train_segment_raw_length: int
    validation_segment_raw_length: int
    test_segment_raw_length: int
    train_segment_time_steps: int
    validation_segment_time_steps: int
    test_segment_time_steps: int
    scenario_train_split_ratio: float
    scenario_validation_split_ratio: float
    scenario_test_split_ratio: float
    scenario_batch_size: int
    shuffle_train_scenarios: bool
    selected_num_stocks: int
    parsed_n: int
    parsed_t: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadedScenarioArrays:
    record: ScenarioFileRecord
    stock_ids: list[str]
    time_index: list[int]
    stock_features_raw: np.ndarray
    market_features_raw: np.ndarray
    stock_returns_raw: np.ndarray


class ScenarioSegmentDataset(Dataset):
    """Dataset returning one full scenario segment per sample."""

    def __init__(self, scenario_segments: list[ScenarioSegmentRecord]) -> None:
        self.scenario_segments = scenario_segments

    def __len__(self) -> int:
        return len(self.scenario_segments)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = self.scenario_segments[index]
        return {
            "scenario_id": item.scenario_id,
            "source_path": item.source_path,
            "split_name": item.split_name,
            "feature_time_indices": torch.from_numpy(item.feature_time_indices.astype(np.int64)),
            "target_time_indices": torch.from_numpy(item.target_time_indices.astype(np.int64)),
            "x_stock": torch.from_numpy(item.x_stock.astype(np.float32)),
            "x_market": torch.from_numpy(item.x_market.astype(np.float32)),
            "r_stock": torch.from_numpy(item.r_stock.astype(np.float32)),
            "stock_indices": torch.from_numpy(item.stock_indices.astype(np.int64)),
        }


class PortfolioPanelDataset:
    """Scenario-only dataset manager.

    Each scenario file represents one path. Train/validation/test scenario groups
    are deterministic file splits. Within each scenario, time is split again and
    each dataset sample is the entire segment, not many windows mixed together.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.scenario_dir = Path(config.scenario_dir)
        if not self.scenario_dir.exists():
            raise FileNotFoundError(f"Scenario directory not found: {self.scenario_dir}")

        self.loaded_stock_feature_columns = list(STOCK_FEATURE_COLUMNS)
        self.loaded_market_feature_columns = list(MARKET_FEATURE_COLUMNS)
        self.ignored_extra_columns: list[str] = []
        self.state = self.scenario_dir.name
        self.stock_scaler = Standardizer()
        self.market_scaler = Standardizer()

        self.train_segment_records: list[ScenarioSegmentRecord] = []
        self.validation_segment_records: list[ScenarioSegmentRecord] = []
        self.test_segment_records: list[ScenarioSegmentRecord] = []
        self._scenario_arrays_cache: dict[str, LoadedScenarioArrays] = {}

        self._discover_scenarios()
        self._fit_standardizers_on_train_scenarios()
        self._materialize_scenario_segments()
        self._build_metadata()

    def _discover_scenarios(self) -> None:
        scenario_glob = self.config.scenario_glob.format(state=self.state)
        matched_paths = sorted(
            self.scenario_dir.glob(scenario_glob),
            key=self._scenario_sort_key,
        )
        records: list[ScenarioFileRecord] = []
        for path in matched_paths:
            info = parse_scenario_file_info(path.name)
            if str(info["state"]) != self.state:
                continue
            records.append(
                ScenarioFileRecord(
                    scenario_id=path.stem,
                    source_path=str(path),
                    state=str(info["state"]),
                    scenario_index=int(info["scenario_index"]),
                    parsed_n=int(info["parsed_n"]),
                    parsed_t=int(info["parsed_t"]),
                )
            )

        expected_total = self.config.expected_total_scenarios
        if len(records) != expected_total:
            raise ValueError(
                f"Expected exactly {expected_total} scenario files in {self.scenario_dir}, "
                f"but found {len(records)} matching '{scenario_glob}'."
            )

        self.scenario_records = records
        self.train_scenario_records = records[: self.config.num_train_scenarios]
        val_start = self.config.num_train_scenarios
        val_end = val_start + self.config.num_validation_scenarios
        self.validation_scenario_records = records[val_start:val_end]
        self.test_scenario_records = records[val_end:]

    @staticmethod
    def _scenario_sort_key(path: Path) -> tuple[str, int]:
        info = parse_scenario_file_info(path.name)
        return str(info["state"]), int(info["scenario_index"])

    def _validate_reference_schema(
        self,
        *,
        record: ScenarioFileRecord,
        stock_ids: list[str],
        time_index: list[int],
    ) -> None:
        if not hasattr(self, "parsed_n"):
            self.parsed_n = record.parsed_n
            self.parsed_t = record.parsed_t
            self.reference_stock_ids = list(stock_ids)
            self.reference_stock_id_to_position = {
                stock_id: index for index, stock_id in enumerate(self.reference_stock_ids)
            }
            self.reference_time_index = list(time_index)
            self.reference_time_index_array = np.asarray(self.reference_time_index, dtype=np.int64)
            self.selected_stock_indices = self._resolve_selected_stock_indices(len(stock_ids))
            self.selected_stock_ids = [
                stock_ids[index] for index in self.selected_stock_indices.tolist()
            ]
            self._resolve_time_segment_lengths(self.parsed_t)
            return

        if record.parsed_n != self.parsed_n or record.parsed_t != self.parsed_t:
            raise ValueError(
                "All scenarios must share the same parsed N/T. "
                f"Expected ({self.parsed_n}, {self.parsed_t}), "
                f"received ({record.parsed_n}, {record.parsed_t}) "
                f"for {record.source_path}."
            )
        if stock_ids != self.reference_stock_ids:
            raise ValueError(
                "All scenarios must share the same stock universe ordering after sorting."
            )
        if time_index != self.reference_time_index:
            raise ValueError("All scenarios must share the same time index ordering after sorting.")

    def _resolve_selected_stock_indices(self, actual_num_stocks: int) -> np.ndarray:
        requested_num_stocks = self.config.num_stocks
        if requested_num_stocks is None:
            return np.arange(actual_num_stocks, dtype=np.int64)
        if requested_num_stocks <= 0:
            raise ValueError("DataConfig.num_stocks must be positive when provided.")
        if requested_num_stocks > actual_num_stocks:
            raise ValueError(
                f"Requested fixed num_stocks={requested_num_stocks}, "
                f"but scenario data only provides {actual_num_stocks} stocks."
            )
        return np.arange(requested_num_stocks, dtype=np.int64)

    def _validate_ratio_sum(
        self,
        *,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
    ) -> None:
        ratio_sum = float(train_ratio) + float(validation_ratio) + float(test_ratio)
        if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "Scenario time split ratios must sum to 1.0. "
                f"Received train={train_ratio}, validation={validation_ratio}, test={test_ratio}, "
                f"sum={ratio_sum:.12f}."
            )
        for name, value in (
            ("scenario_train_split_ratio", train_ratio),
            ("scenario_validation_split_ratio", validation_ratio),
            ("scenario_test_split_ratio", test_ratio),
        ):
            if value <= 0.0:
                raise ValueError(f"DataConfig.{name} must be positive, received {value}.")

    def _resolve_time_segment_lengths(self, total_time_steps: int) -> None:
        self._validate_ratio_sum(
            train_ratio=self.config.scenario_train_split_ratio,
            validation_ratio=self.config.scenario_validation_split_ratio,
            test_ratio=self.config.scenario_test_split_ratio,
        )
        train_end = int(total_time_steps * float(self.config.scenario_train_split_ratio))
        validation_end = int(
            total_time_steps
            * float(
                self.config.scenario_train_split_ratio
                + self.config.scenario_validation_split_ratio
            )
        )

        self.train_segment_start_index = 0
        self.train_segment_end_index = train_end
        self.validation_segment_start_index = train_end
        self.validation_segment_end_index = validation_end
        self.test_segment_start_index = validation_end
        self.test_segment_end_index = total_time_steps

        self.train_segment_raw_length = (
            self.train_segment_end_index - self.train_segment_start_index
        )
        self.validation_segment_raw_length = (
            self.validation_segment_end_index - self.validation_segment_start_index
        )
        self.test_segment_raw_length = self.test_segment_end_index - self.test_segment_start_index

        for split_name, raw_length in (
            ("train", self.train_segment_raw_length),
            ("validation", self.validation_segment_raw_length),
            ("test", self.test_segment_raw_length),
        ):
            if raw_length < 2:
                raise ValueError(
                    f"{split_name}_segment_raw_length={raw_length} is too short. "
                    "Each time split must contain at least 2 raw timestamps so that one-step "
                    "target returns can be formed without future leakage."
                )

        self.train_segment_time_steps = self.train_segment_raw_length - 1
        self.validation_segment_time_steps = self.validation_segment_raw_length - 1
        self.test_segment_time_steps = self.test_segment_raw_length - 1
        self.max_time_steps = max(
            self.train_segment_time_steps,
            self.validation_segment_time_steps,
            self.test_segment_time_steps,
        )

    def _read_scenario_frame(self, source_path: Path) -> pd.DataFrame:
        available_columns = set(pq.read_schema(source_path).names)
        columns = [column_name for column_name in LOADABLE_COLUMNS if column_name in available_columns]
        return pd.read_parquet(source_path, columns=columns)

    def _load_scenario_arrays_uncached(self, scenario_record: ScenarioFileRecord) -> LoadedScenarioArrays:
        source_path = Path(scenario_record.source_path)
        frame = self._read_scenario_frame(source_path)
        header = frame.columns.tolist()
        missing_columns = [column for column in BASE_REQUIRED_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {source_path}: {missing_columns}")
        if not self.ignored_extra_columns:
            self.ignored_extra_columns = []

        has_return_column = OPTIONAL_RETURN_COLUMN in frame.columns
        for column in NUMERIC_COLUMNS:
            if column in frame.columns:
                frame[column] = _coerce_numeric_series(frame[column])
        frame["time_index"] = _parse_time_series(frame["t"])

        if frame.duplicated(["stock_id", "time_index"]).any():
            raise ValueError(f"Scenario contains duplicated (stock_id, t) rows: {source_path}")

        stock_ids = sorted(frame["stock_id"].unique().tolist())
        time_index = sorted(frame["time_index"].unique().tolist())

        if len(stock_ids) != scenario_record.parsed_n:
            raise ValueError(
                f"Parsed N={scenario_record.parsed_n} from file name but CSV contains {len(stock_ids)} stocks."
            )
        if len(time_index) != scenario_record.parsed_t:
            raise ValueError(
                f"Parsed T={scenario_record.parsed_t} from file name but CSV contains {len(time_index)} times."
            )

        self._validate_reference_schema(
            record=scenario_record,
            stock_ids=stock_ids,
            time_index=time_index,
        )

        expected_row_count = scenario_record.parsed_n * scenario_record.parsed_t
        if len(frame) != expected_row_count:
            raise ValueError(f"Scenario is incomplete: {source_path}")
        stock_position = pd.Categorical(
            frame["stock_id"],
            categories=self.reference_stock_ids,
            ordered=True,
        ).codes
        if (stock_position < 0).any():
            raise ValueError("Scenario contains unknown stock IDs compared with the reference universe.")

        time_values = frame["time_index"].to_numpy(dtype=np.int64, copy=False)
        time_position = np.searchsorted(self.reference_time_index_array, time_values)
        if (
            (time_position >= len(self.reference_time_index_array)).any()
            or not np.array_equal(self.reference_time_index_array[time_position], time_values)
        ):
            raise ValueError("Scenario contains unexpected time indices compared with the reference grid.")

        linear_index = stock_position.astype(np.int64) * self.parsed_t + time_position.astype(np.int64)
        coverage = np.bincount(linear_index, minlength=expected_row_count)
        if coverage.shape[0] != expected_row_count or not np.all(coverage == 1):
            raise ValueError(f"Scenario is incomplete after position mapping: {source_path}")

        stock_feature_values = frame[STOCK_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        stock_feature_grid = np.empty((expected_row_count, len(STOCK_FEATURE_COLUMNS)), dtype=np.float32)
        stock_feature_grid[linear_index] = stock_feature_values
        stock_features_raw = stock_feature_grid.reshape(
            self.parsed_n,
            self.parsed_t,
            len(STOCK_FEATURE_COLUMNS),
        ).transpose(1, 0, 2)

        market_values = frame[MARKET_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        market_grid = np.empty((expected_row_count, len(MARKET_FEATURE_COLUMNS)), dtype=np.float32)
        market_grid[linear_index] = market_values
        market_cube = market_grid.reshape(
            self.parsed_n,
            self.parsed_t,
            len(MARKET_FEATURE_COLUMNS),
        ).transpose(1, 0, 2)
        if not np.allclose(market_cube, market_cube[:, :1, :], atol=0.0, rtol=0.0):
            raise ValueError("FF3 factors are not identical across stocks within the same day.")
        market_features_raw = market_cube[:, 0, :]

        if has_return_column:
            return_values = frame[OPTIONAL_RETURN_COLUMN].to_numpy(dtype=np.float32, copy=True)
            return_grid = np.empty((expected_row_count,), dtype=np.float32)
            return_grid[linear_index] = return_values
            stock_returns_raw = return_grid.reshape(self.parsed_n, self.parsed_t).transpose(1, 0)
        else:
            price_array = stock_features_raw[..., -1]
            stock_returns_raw = np.zeros_like(price_array)
            stock_returns_raw[1:] = (price_array[1:] / price_array[:-1]) - 1.0

        return LoadedScenarioArrays(
            record=scenario_record,
            stock_ids=stock_ids,
            time_index=time_index,
            stock_features_raw=stock_features_raw[:, self.selected_stock_indices, :],
            market_features_raw=market_features_raw,
            stock_returns_raw=stock_returns_raw[:, self.selected_stock_indices],
        )

    def _load_scenario_arrays(self, scenario_record: ScenarioFileRecord) -> LoadedScenarioArrays:
        cached = self._scenario_arrays_cache.get(scenario_record.scenario_id)
        if cached is not None:
            return cached
        arrays = self._load_scenario_arrays_uncached(scenario_record)
        self._scenario_arrays_cache[scenario_record.scenario_id] = arrays
        return arrays

    def _fit_standardizers_on_train_scenarios(self) -> None:
        stock_moments = RunningMoments(len(STOCK_FEATURE_COLUMNS))
        market_moments = RunningMoments(len(MARKET_FEATURE_COLUMNS))

        for scenario_record in self.train_scenario_records:
            arrays = self._load_scenario_arrays(scenario_record)
            train_stock_values = arrays.stock_features_raw[
                self.train_segment_start_index : self.train_segment_end_index
            ].reshape(-1, len(STOCK_FEATURE_COLUMNS))
            train_market_values = arrays.market_features_raw[
                self.train_segment_start_index : self.train_segment_end_index
            ]
            stock_moments.update(train_stock_values)
            market_moments.update(train_market_values)

        stock_mean, stock_std = stock_moments.finalize()
        market_mean, market_std = market_moments.finalize()
        self.stock_scaler.set_statistics(stock_mean, stock_std)
        self.market_scaler.set_statistics(market_mean, market_std)

    def _split_bounds_for(self, split_name: str) -> tuple[int, int]:
        if split_name == "train":
            return self.train_segment_start_index, self.train_segment_end_index
        if split_name == "validation":
            return self.validation_segment_start_index, self.validation_segment_end_index
        if split_name == "test":
            return self.test_segment_start_index, self.test_segment_end_index
        raise ValueError(f"Unsupported split_name: {split_name}")

    def _build_segment_record(
        self,
        arrays: LoadedScenarioArrays,
        split_name: str,
    ) -> ScenarioSegmentRecord:
        raw_start, raw_end = self._split_bounds_for(split_name)
        feature_stop = raw_end - 1
        target_start = raw_start + 1

        scaled_stock = self.stock_scaler.transform(
            arrays.stock_features_raw.reshape(-1, len(STOCK_FEATURE_COLUMNS))
        ).reshape(arrays.stock_features_raw.shape)
        scaled_market = self.market_scaler.transform(arrays.market_features_raw)

        x_stock = scaled_stock[raw_start:feature_stop]
        x_market = scaled_market[raw_start:feature_stop]
        r_stock = arrays.stock_returns_raw[target_start:raw_end]
        feature_time_indices = np.asarray(arrays.time_index[raw_start:feature_stop], dtype=np.int64)
        target_time_indices = np.asarray(arrays.time_index[target_start:raw_end], dtype=np.int64)
        stock_indices = np.arange(len(self.selected_stock_ids), dtype=np.int64)

        expected_time_steps = raw_end - raw_start - 1
        assert x_stock.shape == (
            expected_time_steps,
            len(self.selected_stock_ids),
            len(STOCK_FEATURE_COLUMNS),
        )
        assert x_market.shape == (expected_time_steps, len(MARKET_FEATURE_COLUMNS))
        assert r_stock.shape == (expected_time_steps, len(self.selected_stock_ids))
        assert feature_time_indices.shape == target_time_indices.shape == (expected_time_steps,)

        return ScenarioSegmentRecord(
            scenario_id=arrays.record.scenario_id,
            source_path=arrays.record.source_path,
            split_name=split_name,
            feature_time_indices=feature_time_indices,
            target_time_indices=target_time_indices,
            x_stock=x_stock.astype(np.float32),
            x_market=x_market.astype(np.float32),
            r_stock=r_stock.astype(np.float32),
            stock_indices=stock_indices,
        )

    def _materialize_scenario_segments(self) -> None:
        split_map = {
            "train": self.train_scenario_records,
            "validation": self.validation_scenario_records,
            "test": self.test_scenario_records,
        }
        target_lists = {
            "train": self.train_segment_records,
            "validation": self.validation_segment_records,
            "test": self.test_segment_records,
        }

        for split_name, records in split_map.items():
            for scenario_record in records:
                arrays = self._load_scenario_arrays(scenario_record)
                target_lists[split_name].append(self._build_segment_record(arrays, split_name))
        self._scenario_arrays_cache.clear()

    def _build_metadata(self) -> None:
        self.metadata = ScenarioDatasetMetadata(
            scenario_dir=str(self.scenario_dir),
            scenario_glob=self.config.scenario_glob.format(state=self.state),
            state=self.state,
            total_scenarios_found=len(self.scenario_records),
            num_train_scenarios=len(self.train_scenario_records),
            num_validation_scenarios=len(self.validation_scenario_records),
            num_test_scenarios=len(self.test_scenario_records),
            train_scenarios=[record.scenario_id for record in self.train_scenario_records],
            validation_scenarios=[record.scenario_id for record in self.validation_scenario_records],
            test_scenarios=[record.scenario_id for record in self.test_scenario_records],
            total_num_days=self.parsed_t,
            train_segment_start_index=self.train_segment_start_index,
            train_segment_end_index=self.train_segment_end_index - 1,
            validation_segment_start_index=self.validation_segment_start_index,
            validation_segment_end_index=self.validation_segment_end_index - 1,
            test_segment_start_index=self.test_segment_start_index,
            test_segment_end_index=self.test_segment_end_index - 1,
            train_segment_raw_length=self.train_segment_raw_length,
            validation_segment_raw_length=self.validation_segment_raw_length,
            test_segment_raw_length=self.test_segment_raw_length,
            train_segment_time_steps=self.train_segment_time_steps,
            validation_segment_time_steps=self.validation_segment_time_steps,
            test_segment_time_steps=self.test_segment_time_steps,
            scenario_train_split_ratio=float(self.config.scenario_train_split_ratio),
            scenario_validation_split_ratio=float(self.config.scenario_validation_split_ratio),
            scenario_test_split_ratio=float(self.config.scenario_test_split_ratio),
            scenario_batch_size=int(self.config.scenario_batch_size),
            shuffle_train_scenarios=bool(self.config.shuffle_train_scenarios),
            selected_num_stocks=len(self.selected_stock_ids),
            parsed_n=self.parsed_n,
            parsed_t=self.parsed_t,
        )

    @property
    def num_stocks(self) -> int:
        return len(self.selected_stock_ids)

    @property
    def num_times(self) -> int:
        return self.parsed_t

    def build_train_validation_test_datasets(
        self,
    ) -> tuple[ScenarioSegmentDataset, ScenarioSegmentDataset, ScenarioSegmentDataset]:
        return (
            ScenarioSegmentDataset(list(self.train_segment_records)),
            ScenarioSegmentDataset(list(self.validation_segment_records)),
            ScenarioSegmentDataset(list(self.test_segment_records)),
        )

    def build_train_validation_backtest_datasets(
        self,
    ) -> tuple[ScenarioSegmentDataset, ScenarioSegmentDataset, ScenarioSegmentDataset]:
        return self.build_train_validation_test_datasets()

    def get_split_dataset(self, split_name: str) -> ScenarioSegmentDataset:
        if split_name == "train":
            return ScenarioSegmentDataset(list(self.train_segment_records))
        if split_name == "validation":
            return ScenarioSegmentDataset(list(self.validation_segment_records))
        if split_name == "test":
            return ScenarioSegmentDataset(list(self.test_segment_records))
        raise ValueError(f"Unsupported split_name: {split_name}")

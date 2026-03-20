from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_attention.config import DataConfig
from portfolio_attention.dataset import PortfolioPanelDataset, parse_panel_dimensions


def write_panel_csv(path: Path, num_stocks: int = 4, num_times: int = 81) -> Path:
    rows = []
    for stock_idx in range(num_stocks):
        price = 100.0 + stock_idx
        for time_idx in range(num_times):
            price = price * (1.0 + 0.001 * (stock_idx + 1) + 0.0001 * time_idx)
            rows.append(
                {
                    "stock_id": f"stock_{stock_idx:03d}",
                    "t": f"t_{time_idx}",
                    "characteristic_1": stock_idx + time_idx * 0.1,
                    "characteristic_2": stock_idx * 2 + time_idx * 0.2,
                    "characteristic_3": stock_idx * 3 + time_idx * 0.3,
                    "MKT": 0.01 * time_idx,
                    "SMB": 0.02 * time_idx,
                    "HML": 0.03 * time_idx,
                    "price": price,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_parse_panel_dimensions() -> None:
    assert parse_panel_dimensions("bull_4860_81_panel_long.csv") == (4860, 81)
    assert parse_panel_dimensions("custom_prefix_12_99_panel_long.csv") == (12, 99)


def test_fixed_schema_validation(tmp_path: Path) -> None:
    csv_path = write_panel_csv(tmp_path / "mini_4_81_panel_long.csv")
    frame = pd.read_csv(csv_path)
    frame = frame.drop(columns=["HML"])
    bad_path = tmp_path / "mini_missing_4_81_panel_long.csv"
    frame.to_csv(bad_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        PortfolioPanelDataset(DataConfig(csv_path=bad_path))


def test_ff3_consistency_validation(tmp_path: Path) -> None:
    csv_path = write_panel_csv(tmp_path / "mini_4_81_panel_long.csv")
    frame = pd.read_csv(csv_path)
    frame.loc[(frame["stock_id"] == "stock_001") & (frame["t"] == "t_10"), "MKT"] = 999.0
    bad_path = tmp_path / "mini_bad_ff3_4_81_panel_long.csv"
    frame.to_csv(bad_path, index=False)

    with pytest.raises(ValueError, match="FF3 factors"):
        PortfolioPanelDataset(DataConfig(csv_path=bad_path))


def test_train_only_scaler_and_tensor_shapes(tmp_path: Path) -> None:
    csv_path = write_panel_csv(tmp_path / "mini_4_81_panel_long.csv")
    dataset = PortfolioPanelDataset(DataConfig(csv_path=csv_path))
    batch = dataset.get_analysis_batch()

    manual_train_mean = dataset.stock_features_raw[:, : dataset.train_days, :].reshape(-1, 4).mean(axis=0)
    assert np.allclose(dataset.stock_scaler.mean, manual_train_mean)
    assert batch["x_stock"].shape == (1, 4, 60, 4)
    assert batch["x_market"].shape == (1, 60, 3)
    assert batch["r_stock"].shape == (1, 4)


def test_t81_honest_window_counts_and_cross_boundary_window(tmp_path: Path) -> None:
    csv_path = write_panel_csv(tmp_path / "mini_4_81_panel_long.csv")
    dataset = PortfolioPanelDataset(DataConfig(csv_path=csv_path))
    window = dataset.get_analysis_window()

    assert dataset.metadata.legal_train_windows == 0
    assert dataset.metadata.legal_test_windows == 0
    assert dataset.metadata.available_analysis_windows == 1

    entry_price = dataset.price_array[0, 60]
    exit_price = dataset.price_array[0, 79]
    expected_return = (exit_price / entry_price) - 1.0
    assert np.isclose(window["r_stock"][0, 0], expected_return)


def test_fixed_num_stocks_selects_exact_count(tmp_path: Path) -> None:
    csv_path = write_panel_csv(tmp_path / "mini_4_81_panel_long.csv")
    dataset = PortfolioPanelDataset(DataConfig(csv_path=csv_path, num_stocks=3))

    assert dataset.num_stocks == 3
    assert dataset.selected_stock_ids == ["stock_000", "stock_001", "stock_002"]
    assert dataset.metadata.selected_num_stocks == 3


def test_fixed_num_stocks_requires_sufficient_data(tmp_path: Path) -> None:
    csv_path = write_panel_csv(tmp_path / "mini_4_81_panel_long.csv")

    with pytest.raises(ValueError, match="Requested fixed num_stocks=5"):
        PortfolioPanelDataset(DataConfig(csv_path=csv_path, num_stocks=5))

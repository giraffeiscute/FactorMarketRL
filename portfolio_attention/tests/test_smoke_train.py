from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from portfolio_attention.config import DataConfig, ModelConfig, PathsConfig, TrainConfig
from portfolio_attention.evaluate import enrich_top_k_positions, run_diagnostic_evaluation
from portfolio_attention.train import run_diagnostic_training


def write_panel_csv(path: Path, num_stocks: int = 8, num_times: int = 81, include_aux: bool = True) -> Path:
    rows = []
    for stock_idx in range(num_stocks):
        price = 100.0 + stock_idx
        for time_idx in range(num_times):
            price = price * (1.0 + 0.001 * (stock_idx + 1) + 0.0002 * time_idx)
            row = {
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
            if include_aux:
                row["mu"] = f"mu_{stock_idx}_{time_idx}"
                row["alpha"] = f"alpha_{stock_idx}_{time_idx}"
                row["epsilon_variance"] = f"eps_{stock_idx}_{time_idx}"
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_cpu_only_smoke_train_and_evaluate(tmp_path: Path) -> None:
    csv_path = write_panel_csv(tmp_path / "mini_8_81_panel_long.csv")
    paths = PathsConfig(output_root=tmp_path / "outputs")
    data_config = DataConfig(csv_path=csv_path)
    train_config = TrainConfig(device="cpu", diagnostic_steps=1, max_stocks=8)

    metrics = run_diagnostic_training(data_config, ModelConfig(), train_config, paths)
    evaluation = run_diagnostic_evaluation(data_config, paths, device_name="cpu")

    assert metrics["diagnostic_only"] is True
    assert evaluation["diagnostic_only"] is True
    assert (paths.checkpoints_dir / "diagnostic_last.pt").exists()
    assert (paths.metrics_dir / "diagnostic_metrics.json").exists()
    prediction_json = paths.predictions_dir / "diagnostic_predictions.json"
    metrics_json = paths.metrics_dir / "evaluation_metrics.json"
    assert prediction_json.exists()
    assert metrics_json.exists()
    exported = json.loads(prediction_json.read_text(encoding="utf-8"))
    metrics_exported = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert evaluation["source_path"] == "mini_8_81"
    assert "checkpoint_path" not in evaluation
    assert exported["source_path"] == "mini_8_81"
    assert sorted(exported.keys()) == sorted(
        [
            "average_cash_weight",
            "average_portfolio_return",
            "cash_weight",
            "device",
            "diagnostic_only",
            "metadata",
            "portfolio_return",
            "sharpe_like",
            "source_path",
            "top_k_stock_weights",
        ]
    )
    assert len(metrics_exported["all_stock_weights"]) == 8
    assert len(metrics_exported["allocation_groups"]) >= 1
    assert len(metrics_exported["allocation_groups_top10_plus_others"]) >= 1
    assert Path(metrics_exported["allocation_pie_chart"]).exists()
    assert Path(metrics_exported["allocation_bar_chart"]).exists()
    assert Path(metrics_exported["allocation_pie_chart"]).parent == paths.outputs_dir
    assert Path(metrics_exported["allocation_bar_chart"]).parent == paths.outputs_dir
    assert Path(metrics_exported["all_stock_weights_csv"]).exists()
    first_stock = evaluation["top_k_stock_weights"][0]["stock_id"]
    stock_idx = int(str(first_stock).split("_")[1])
    assert evaluation["top_k_stock_weights"][0]["mu"] == f"mu_{stock_idx}_60"
    assert evaluation["top_k_stock_weights"][0]["alpha"] == f"alpha_{stock_idx}_60"
    assert evaluation["top_k_stock_weights"][0]["epsilon_variance"] == f"eps_{stock_idx}_60"
    all_stock_weights_csv = pd.read_csv(metrics_exported["all_stock_weights_csv"])
    assert len(all_stock_weights_csv) == 8
    assert all_stock_weights_csv.loc[0, "weight"] >= all_stock_weights_csv.loc[len(all_stock_weights_csv) - 1, "weight"]


def test_export_rows_require_aux_columns(tmp_path: Path) -> None:
    csv_path = write_panel_csv(tmp_path / "mini_8_81_panel_long.csv", include_aux=False)

    with pytest.raises(ValueError, match="Missing"):
        enrich_top_k_positions(
            source_csv_path=csv_path,
            metadata={
                "analysis_entry_day": 61,
                "analysis_exit_day": 80,
                "analysis_only": True,
                "available_analysis_windows": 1,
                "csv_unique_stocks": 8,
                "csv_unique_times": 81,
                "effective_num_stocks": 8,
                "effective_time_steps": 81,
                "legal_test_windows": 0,
                "legal_train_windows": 0,
                "lookback": 60,
                "parsed_n": 8,
                "parsed_t": 81,
                "train_days": 60,
                "test_days": 21,
            },
            top_positions=[{"stock_id": "stock_000", "weight": 0.5}],
        )


def test_export_rows_require_single_match(tmp_path: Path) -> None:
    csv_path = write_panel_csv(tmp_path / "mini_8_81_panel_long.csv")
    frame = pd.read_csv(csv_path)
    duplicate_row = frame[(frame["stock_id"] == "stock_000") & (frame["t"] == "t_60")].copy()
    frame = pd.concat([frame, duplicate_row], ignore_index=True)
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="multiple source rows"):
        enrich_top_k_positions(
            source_csv_path=csv_path,
            metadata={
                "analysis_entry_day": 61,
                "analysis_exit_day": 80,
                "analysis_only": True,
                "available_analysis_windows": 1,
                "csv_unique_stocks": 8,
                "csv_unique_times": 81,
                "effective_num_stocks": 8,
                "effective_time_steps": 81,
                "legal_test_windows": 0,
                "legal_train_windows": 0,
                "lookback": 60,
                "parsed_n": 8,
                "parsed_t": 81,
                "train_days": 60,
                "test_days": 21,
            },
            top_positions=[{"stock_id": "stock_000", "weight": 0.5}],
        )

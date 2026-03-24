from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from portfolio_attention.config import DataConfig, ModelConfig, PathsConfig, TrainConfig
from portfolio_attention.evaluate import enrich_top_k_positions, run_diagnostic_evaluation
from portfolio_attention.train import (
    build_arg_parser,
    resolve_runtime_configs_from_args,
    run_diagnostic_training,
    run_training,
)


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
    data_config = DataConfig(csv_path=csv_path, num_stocks=8, analysis_horizon_days=2)
    train_config = TrainConfig(
        device="cpu",
        diagnostic_steps=1,
        batch_size=16,
        num_epochs=30,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        early_stopping_patience=5,
    )

    metrics = run_diagnostic_training(data_config, ModelConfig(), train_config, paths)
    evaluation = run_diagnostic_evaluation(data_config, paths, device_name="cpu")

    assert metrics["diagnostic_only"] is True
    assert evaluation["diagnostic_only"] is True
    assert (paths.checkpoints_dir / train_config.checkpoint_name).exists()
    assert (paths.metrics_dir / "diagnostic_metrics.json").exists()
    state_id = "mini_8_81"
    prediction_json = paths.predictions_dir / f"{state_id}_dsr_diagnostic_predictions.json"
    metrics_json = paths.metrics_dir / "evaluation_metrics.json"
    assert prediction_json.exists()
    assert metrics_json.exists()
    exported = json.loads(prediction_json.read_text(encoding="utf-8"))
    metrics_exported = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert evaluation["source_path"] == "mini_8_81"
    assert "checkpoint_path" not in evaluation
    assert exported["source_path"] == "mini_8_81"
    required_keys = {
        "average_cash_weight",
        "average_portfolio_return",
        "cash_weight",
        "device",
        "diagnostic_only",
        "metadata",
        "portfolio_return",
        "source_path",
        "train_config",
        "top_k_stock_weights",
    }
    assert required_keys.issubset(set(exported.keys()))
    assert exported["train_config"] == {
        "batch_size": 16,
        "num_epochs": 30,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "early_stopping_patience": 5,
    }
    assert len(metrics_exported["all_stock_weights"]) == 8
    assert len(metrics_exported["allocation_groups"]) >= 1
    assert len(metrics_exported["allocation_groups_top_n_plus_others"]) >= 1
    assert Path(metrics_exported["allocation_pie_chart"]).exists()
    assert Path(metrics_exported["allocation_bar_chart"]).exists()
    assert Path(metrics_exported["allocation_pie_chart"]).parent == paths.outputs_dir
    assert Path(metrics_exported["allocation_bar_chart"]).parent == paths.outputs_dir
    assert Path(metrics_exported["all_stock_weights_csv"]).exists()
    first_stock = evaluation["top_k_stock_weights"][0]["stock_id"]
    stock_idx = int(str(first_stock).split("_")[1])
    assert evaluation["top_k_stock_weights"][0]["mu"] == f"mu_{stock_idx}_79"
    assert evaluation["top_k_stock_weights"][0]["alpha"] == f"alpha_{stock_idx}_79"
    assert evaluation["top_k_stock_weights"][0]["epsilon_variance"] == f"eps_{stock_idx}_79"
    all_stock_weights_csv = pd.read_csv(metrics_exported["all_stock_weights_csv"])
    assert len(all_stock_weights_csv) == 8
    assert all_stock_weights_csv.loc[0, "weight"] >= all_stock_weights_csv.loc[len(all_stock_weights_csv) - 1, "weight"]
    diagnostic_checkpoint = torch.load(
        paths.checkpoints_dir / train_config.checkpoint_name,
        map_location="cpu",
        weights_only=False,
    )
    assert "num_stocks" not in diagnostic_checkpoint["model_config"]


def test_epoch_train_mode_saves_best_and_last_checkpoints(tmp_path: Path) -> None:
    csv_path = write_panel_csv(tmp_path / "mini_8_100_panel_long.csv", num_times=100)
    paths = PathsConfig(output_root=tmp_path / "outputs")
    train_config = TrainConfig(
        mode="train",
        device="cpu",
        batch_size=16,
        num_epochs=4,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        early_stopping_patience=2,
    )
    data_config = DataConfig(csv_path=csv_path, num_stocks=8, analysis_horizon_days=2)

    metrics = run_training(data_config, ModelConfig(), train_config, paths)

    assert metrics["diagnostic_only"] is False
    assert metrics["mode"] == "train"
    assert metrics["batch_size"] == 16
    assert metrics["num_epochs_requested"] == 4
    assert metrics["train_window_count"] >= 1
    assert metrics["validation_window_count"] >= 1
    assert metrics["epochs_completed"] <= 4
    assert metrics["best_epoch"] <= metrics["epochs_completed"]
    assert (paths.checkpoints_dir / train_config.train_best_checkpoint_name).exists()
    assert (paths.checkpoints_dir / train_config.train_last_checkpoint_name).exists()
    assert (paths.metrics_dir / "train_metrics.json").exists()

    best_checkpoint = torch.load(
        paths.checkpoints_dir / train_config.train_best_checkpoint_name,
        map_location="cpu",
        weights_only=False,
    )
    assert "num_stocks" not in best_checkpoint["model_config"]
    assert best_checkpoint["train_config"]["batch_size"] == 16
    assert best_checkpoint["train_config"]["weight_decay"] == pytest.approx(1e-4)
    assert best_checkpoint["train_config"]["grad_clip_norm"] == pytest.approx(1.0)
    assert best_checkpoint["train_config"]["early_stopping_patience"] == 2


def test_cli_defaults_resolve_from_config_objects() -> None:
    args = build_arg_parser().parse_args([])
    data_config, train_config = resolve_runtime_configs_from_args(args)

    assert data_config == DataConfig()
    assert train_config == TrainConfig()


def test_cli_explicit_overrides_replace_config_values() -> None:
    args = build_arg_parser().parse_args(
        [
            "--mode",
            "diagnostic",
            "--seed",
            "7",
            "--device",
            "cpu",
            "--loss",
            "sharpe",
            "--diagnostic-steps",
            "3",
            "--num-stocks",
            "12",
            "--batch-size",
            "8",
            "--num-epochs",
            "9",
            "--weight-decay",
            "0.123",
            "--grad-clip-norm",
            "2.5",
            "--early-stopping-patience",
            "4",
        ]
    )
    data_config, train_config = resolve_runtime_configs_from_args(args)

    assert data_config.num_stocks == 12
    assert train_config.mode == "diagnostic"
    assert train_config.seed == 7
    assert train_config.device == "cpu"
    assert train_config.loss_name == "sharpe"
    assert train_config.diagnostic_steps == 3
    assert train_config.batch_size == 8
    assert train_config.num_epochs == 9
    assert train_config.weight_decay == pytest.approx(0.123)
    assert train_config.grad_clip_norm == pytest.approx(2.5)
    assert train_config.early_stopping_patience == 4


def test_export_rows_require_aux_columns(tmp_path: Path) -> None:
    csv_path = write_panel_csv(tmp_path / "mini_8_81_panel_long.csv", include_aux=False)

    with pytest.raises(ValueError, match="Missing"):
        enrich_top_k_positions(
            source_csv_path=csv_path,
                metadata={
                    "backtest_horizon_start_index": 60,
                    "model_lookback": 60,
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
                    "backtest_horizon_start_index": 60,
                    "model_lookback": 60,
                },
            top_positions=[{"stock_id": "stock_000", "weight": 0.5}],
        )

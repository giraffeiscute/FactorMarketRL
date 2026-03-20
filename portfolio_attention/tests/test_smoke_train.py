from __future__ import annotations

from pathlib import Path

import pandas as pd

from portfolio_attention.config import DataConfig, ModelConfig, PathsConfig, TrainConfig
from portfolio_attention.evaluate import run_diagnostic_evaluation
from portfolio_attention.train import run_diagnostic_training


def write_panel_csv(path: Path, num_stocks: int = 8, num_times: int = 81) -> Path:
    rows = []
    for stock_idx in range(num_stocks):
        price = 100.0 + stock_idx
        for time_idx in range(num_times):
            price = price * (1.0 + 0.001 * (stock_idx + 1) + 0.0002 * time_idx)
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
    assert (paths.predictions_dir / "diagnostic_predictions.json").exists()

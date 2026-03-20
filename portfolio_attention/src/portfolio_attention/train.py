"""Diagnostic training entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio_attention.config import DataConfig, ModelConfig, PathsConfig, TrainConfig
    from portfolio_attention.dataset import PortfolioPanelDataset
    from portfolio_attention.losses import build_loss
    from portfolio_attention.model import PortfolioAttentionModel
    from portfolio_attention.utils import (
        append_log,
        ensure_output_dirs,
        resolve_device,
        save_json,
        set_seed,
    )
else:
    from .config import DataConfig, ModelConfig, PathsConfig, TrainConfig
    from .dataset import PortfolioPanelDataset
    from .losses import build_loss
    from .model import PortfolioAttentionModel
    from .utils import append_log, ensure_output_dirs, resolve_device, save_json, set_seed


def _serialize_config(config: object) -> dict:
    serialized = asdict(config)  # type: ignore[arg-type]
    for key, value in list(serialized.items()):
        if isinstance(value, Path):
            serialized[key] = str(value)
    return serialized


def run_diagnostic_training(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    paths: PathsConfig,
) -> dict:
    set_seed(train_config.seed)
    ensure_output_dirs(paths)
    device = resolve_device(train_config.device)

    dataset = PortfolioPanelDataset(
        replace(data_config, max_stocks=train_config.max_stocks or data_config.max_stocks)
    )
    resolved_model_config = replace(model_config, num_stocks=dataset.num_stocks, lookback=data_config.lookback)
    model = PortfolioAttentionModel(resolved_model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    batch = dataset.get_analysis_batch(device=device)

    metrics: dict[str, float | int | bool | str | dict] = {
        "mode": train_config.mode,
        "device": str(device),
        "diagnostic_only": True,
        "legal_train_windows": dataset.metadata.legal_train_windows,
        "legal_test_windows": dataset.metadata.legal_test_windows,
        "analysis_windows": dataset.metadata.available_analysis_windows,
        "effective_num_stocks": dataset.metadata.effective_num_stocks,
    }

    log_path = paths.logs_dir / "train.log"
    append_log(
        log_path,
        "Running diagnostic mode because the fixed T=81 sample definition yields 0 legal train windows and 0 legal test windows.",
    )

    loss_value = None
    portfolio_return = None
    model.train()
    for step in range(train_config.diagnostic_steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["x_stock"],
            batch["x_market"],
            batch["stock_indices"],
            target_returns=batch["r_stock"],
        )
        portfolio_return = outputs["portfolio_return"]
        if portfolio_return is None:
            raise RuntimeError("Diagnostic batch must provide target returns.")
        loss = build_loss(train_config.loss_name, portfolio_return)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        append_log(
            log_path,
            f"diagnostic_step={step} loss={loss_value:.8f} portfolio_return={float(portfolio_return.mean().detach().cpu().item()):.8f}",
        )

    checkpoint_path = paths.checkpoints_dir / train_config.checkpoint_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": resolved_model_config.as_dict(),
            "data_config": _serialize_config(data_config),
            "train_config": _serialize_config(train_config),
            "metadata": dataset.metadata.as_dict(),
        },
        checkpoint_path,
    )

    if portfolio_return is None or loss_value is None:
        raise RuntimeError("Diagnostic loop did not produce outputs.")

    metrics.update(
        {
            "loss_name": train_config.loss_name,
            "final_loss": loss_value,
            "portfolio_return": float(portfolio_return.mean().detach().cpu().item()),
            "checkpoint_path": str(checkpoint_path),
            "metadata": dataset.metadata.as_dict(),
        }
    )
    save_json(metrics, paths.metrics_dir / "diagnostic_metrics.json")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run diagnostic training for portfolio_attention.")
    parser.add_argument("--mode", default="diagnostic")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--loss", default="return", choices=["return", "sharpe"])
    parser.add_argument("--diagnostic-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-stocks", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = PathsConfig()
    default_data_config = DataConfig()
    data_config = DataConfig(
        csv_path=args.data_path or default_data_config.csv_path,
        max_stocks=args.max_stocks,
    )
    train_config = TrainConfig(
        mode=args.mode,
        device=args.device,
        diagnostic_steps=args.diagnostic_steps,
        seed=args.seed,
        loss_name=args.loss,
        max_stocks=args.max_stocks,
    )
    metrics = run_diagnostic_training(data_config, ModelConfig(), train_config, paths)
    print(metrics)


if __name__ == "__main__":
    main()

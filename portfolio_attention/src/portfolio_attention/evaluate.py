"""Diagnostic evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio_attention.config import DataConfig, ModelConfig, PathsConfig
    from portfolio_attention.dataset import PortfolioPanelDataset
    from portfolio_attention.losses import sharpe_loss
    from portfolio_attention.model import PortfolioAttentionModel
    from portfolio_attention.utils import ensure_output_dirs, resolve_device, save_json
else:
    from .config import DataConfig, ModelConfig, PathsConfig
    from .dataset import PortfolioPanelDataset
    from .losses import sharpe_loss
    from .model import PortfolioAttentionModel
    from .utils import ensure_output_dirs, resolve_device, save_json


def run_diagnostic_evaluation(
    data_config: DataConfig,
    paths: PathsConfig,
    checkpoint_path: Path | None = None,
    device_name: str = "auto",
    top_k: int = 5,
) -> dict:
    ensure_output_dirs(paths)
    device = resolve_device(device_name)
    dataset = PortfolioPanelDataset(data_config)
    batch = dataset.get_analysis_batch(device=device)

    resolved_checkpoint = checkpoint_path or (paths.checkpoints_dir / "diagnostic_last.pt")
    checkpoint = torch.load(resolved_checkpoint, map_location=device, weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = PortfolioAttentionModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        outputs = model(
            batch["x_stock"],
            batch["x_market"],
            batch["stock_indices"],
            target_returns=batch["r_stock"],
        )

    stock_weights = outputs["stock_weights"][0].detach().cpu()
    cash_weight = float(outputs["cash_weight"][0].detach().cpu().item())
    portfolio_return = float(outputs["portfolio_return"][0].detach().cpu().item())
    sharpe_like = float((-sharpe_loss(outputs["portfolio_return"]).detach().cpu().item()))
    top_k = min(top_k, dataset.num_stocks)
    top_values, top_indices = torch.topk(stock_weights, k=top_k)
    top_positions = [
        {
            "stock_id": dataset.effective_stock_ids[int(index)],
            "weight": float(weight.item()),
        }
        for weight, index in zip(top_values, top_indices)
    ]

    payload = {
        "diagnostic_only": True,
        "device": str(device),
        "checkpoint_path": str(resolved_checkpoint),
        "portfolio_return": portfolio_return,
        "average_portfolio_return": portfolio_return,
        "cash_weight": cash_weight,
        "average_cash_weight": cash_weight,
        "sharpe_like": sharpe_like,
        "top_k_stock_weights": top_positions,
        "metadata": dataset.metadata.as_dict(),
    }
    save_json(payload, paths.predictions_dir / "diagnostic_predictions.json")
    save_json(payload, paths.metrics_dir / "evaluation_metrics.json")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run diagnostic evaluation for portfolio_attention.")
    parser.add_argument("--mode", default="diagnostic")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=5)
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
    payload = run_diagnostic_evaluation(
        data_config=data_config,
        paths=paths,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
        top_k=args.top_k,
    )
    print(payload)


if __name__ == "__main__":
    main()

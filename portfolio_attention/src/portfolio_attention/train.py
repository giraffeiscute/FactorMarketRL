"""Training entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

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


TRAIN_BEST_CHECKPOINT_NAME = "train_best.pt"
TRAIN_LAST_CHECKPOINT_NAME = "train_last.pt"


def _serialize_config(config: object) -> dict:
    serialized = asdict(config)  # type: ignore[arg-type]
    for key, value in list(serialized.items()):
        if isinstance(value, Path):
            serialized[key] = str(value)
    return serialized


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _resolve_model_config(model_config: ModelConfig, data_config: DataConfig) -> ModelConfig:
    return replace(model_config, lookback=data_config.lookback)


def _run_loss_step(
    model: PortfolioAttentionModel,
    batch: dict[str, torch.Tensor],
    loss_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(
        batch["x_stock"],
        batch["x_market"],
        batch["stock_indices"],
        target_returns=batch["r_stock"],
    )
    portfolio_return = outputs["portfolio_return"]
    if portfolio_return is None:
        raise RuntimeError("Training batch must provide target returns.")
    loss = build_loss(loss_name, portfolio_return)
    return loss, portfolio_return


def _build_checkpoint_payload(
    *,
    model: PortfolioAttentionModel,
    optimizer: torch.optim.Optimizer,
    resolved_model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    dataset: PortfolioPanelDataset,
    epoch: int | None,
    best_val_loss: float | None,
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": resolved_model_config.as_dict(),
        "data_config": _serialize_config(data_config),
        "train_config": _serialize_config(train_config),
        "metadata": dataset.metadata.as_dict(),
    }
    if extra_metrics:
        payload["metrics"] = extra_metrics
    return payload


@torch.no_grad()
def _evaluate_epoch(
    model: PortfolioAttentionModel,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_return = 0.0
    total_samples = 0

    for raw_batch in loader:
        batch = _move_batch_to_device(raw_batch, device)
        loss, portfolio_return = _run_loss_step(model, batch, loss_name)
        batch_size = int(batch["x_stock"].shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_return += float(portfolio_return.mean().detach().cpu().item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Validation loader produced no samples.")

    return total_loss / total_samples, total_return / total_samples


def run_diagnostic_training(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    paths: PathsConfig,
) -> dict:
    set_seed(train_config.seed)
    ensure_output_dirs(paths)
    device = resolve_device(train_config.device)

    dataset = PortfolioPanelDataset(data_config)
    resolved_model_config = _resolve_model_config(model_config, data_config)
    model = PortfolioAttentionModel(resolved_model_config, num_stocks=dataset.num_stocks).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    batch = dataset.get_analysis_batch(device=device)

    metrics: dict[str, float | int | bool | str | dict] = {
        "mode": "diagnostic",
        "device": str(device),
        "diagnostic_only": True,
        "loaded_feature_columns": {
            "stock": dataset.loaded_stock_feature_columns,
            "market": dataset.loaded_market_feature_columns,
        },
        "legal_train_windows": dataset.metadata.legal_train_windows,
        "legal_test_windows": dataset.metadata.legal_test_windows,
        "analysis_windows": dataset.metadata.available_analysis_windows,
        "selected_num_stocks": dataset.metadata.selected_num_stocks,
    }

    log_path = paths.logs_dir / "train.log"
    append_log(
        log_path,
        (
            "Loaded feature columns successfully: "
            f"stock={dataset.loaded_stock_feature_columns} "
            f"market={dataset.loaded_market_feature_columns}"
        ),
    )
    append_log(
        log_path,
        "Running diagnostic mode because the fixed T=81 sample definition yields 0 legal train windows and 0 legal test windows.",
    )

    loss_value = None
    portfolio_return = None
    model.train()
    for step in range(train_config.diagnostic_steps):
        optimizer.zero_grad(set_to_none=True)
        loss, portfolio_return = _run_loss_step(model, batch, train_config.loss_name)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        append_log(
            log_path,
            f"diagnostic_step={step} loss={loss_value:.8f} portfolio_return={float(portfolio_return.mean().detach().cpu().item()):.8f}",
        )

    checkpoint_path = paths.checkpoints_dir / train_config.checkpoint_name
    torch.save(
        _build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            resolved_model_config=resolved_model_config,
            data_config=data_config,
            train_config=train_config,
            dataset=dataset,
            epoch=None,
            best_val_loss=None,
        ),
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


def run_epoch_training(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    paths: PathsConfig,
) -> dict:
    set_seed(train_config.seed)
    ensure_output_dirs(paths)
    device = resolve_device(train_config.device)

    dataset = PortfolioPanelDataset(data_config)
    train_dataset, val_dataset = dataset.build_train_val_datasets()
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            "Train mode requires at least one train window and one validation window. "
            "Use diagnostic mode for single-window analysis."
        )

    resolved_model_config = _resolve_model_config(model_config, data_config)
    model = PortfolioAttentionModel(resolved_model_config, num_stocks=dataset.num_stocks).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    generator = torch.Generator()
    generator.manual_seed(train_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
    )

    log_path = paths.logs_dir / "train.log"
    append_log(
        log_path,
        (
            "Loaded feature columns successfully: "
            f"stock={dataset.loaded_stock_feature_columns} "
            f"market={dataset.loaded_market_feature_columns}"
        ),
    )
    append_log(
        log_path,
        (
            "Running epoch-based train mode with "
            f"train_windows={len(train_dataset)} val_windows={len(val_dataset)} "
            f"batch_size={train_config.batch_size} num_epochs={train_config.num_epochs}."
        ),
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_completed = 0
    history: list[dict[str, float | int]] = []
    best_checkpoint_path = paths.checkpoints_dir / TRAIN_BEST_CHECKPOINT_NAME
    last_checkpoint_path = paths.checkpoints_dir / TRAIN_LAST_CHECKPOINT_NAME

    for epoch in range(1, train_config.num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_return = 0.0
        total_train_samples = 0

        for raw_batch in train_loader:
            batch = _move_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, portfolio_return = _run_loss_step(model, batch, train_config.loss_name)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
            optimizer.step()

            batch_size = int(batch["x_stock"].shape[0])
            total_train_loss += float(loss.detach().cpu().item()) * batch_size
            total_train_return += float(portfolio_return.mean().detach().cpu().item()) * batch_size
            total_train_samples += batch_size

        if total_train_samples == 0:
            raise RuntimeError("Train loader produced no samples.")

        train_loss = total_train_loss / total_train_samples
        train_return = total_train_return / total_train_samples
        val_loss, val_return = _evaluate_epoch(model, val_loader, device, train_config.loss_name)

        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_portfolio_return": train_return,
            "val_loss": val_loss,
            "val_portfolio_return": val_return,
        }
        history.append(epoch_metrics)
        epochs_completed = epoch

        append_log(
            log_path,
            (
                f"epoch={epoch} train_loss={train_loss:.8f} "
                f"train_portfolio_return={train_return:.8f} "
                f"val_loss={val_loss:.8f} val_portfolio_return={val_return:.8f}"
            ),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                _build_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    resolved_model_config=resolved_model_config,
                    data_config=data_config,
                    train_config=train_config,
                    dataset=dataset,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    extra_metrics=epoch_metrics,
                ),
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        torch.save(
            _build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                resolved_model_config=resolved_model_config,
                data_config=data_config,
                train_config=train_config,
                dataset=dataset,
                epoch=epoch,
                best_val_loss=best_val_loss,
                extra_metrics=epoch_metrics,
            ),
            last_checkpoint_path,
        )

        if epochs_without_improvement >= train_config.early_stopping_patience:
            append_log(
                log_path,
                (
                    "Early stopping triggered with "
                    f"patience={train_config.early_stopping_patience} at epoch={epoch}."
                ),
            )
            break

    if best_epoch == 0:
        raise RuntimeError("Train loop did not record a best checkpoint.")

    metrics: dict[str, Any] = {
        "mode": "train",
        "device": str(device),
        "diagnostic_only": False,
        "loaded_feature_columns": {
            "stock": dataset.loaded_stock_feature_columns,
            "market": dataset.loaded_market_feature_columns,
        },
        "loss_name": train_config.loss_name,
        "batch_size": train_config.batch_size,
        "num_epochs_requested": train_config.num_epochs,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "last_train_loss": history[-1]["train_loss"],
        "last_val_loss": history[-1]["val_loss"],
        "train_window_count": len(train_dataset),
        "val_window_count": len(val_dataset),
        "early_stopping_patience": train_config.early_stopping_patience,
        "stopped_early": epochs_completed < train_config.num_epochs,
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "metadata": dataset.metadata.as_dict(),
        "history": history,
    }
    save_json(metrics, paths.metrics_dir / "train_metrics.json")
    return metrics


def run_training(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    paths: PathsConfig,
) -> dict:
    normalized_mode = train_config.mode.lower()
    if normalized_mode == "diagnostic":
        return run_diagnostic_training(data_config, model_config, train_config, paths)
    if normalized_mode == "train":
        return run_epoch_training(data_config, model_config, train_config, paths)
    raise ValueError(f"Unsupported mode: {train_config.mode}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run training for portfolio_attention.")
    parser.add_argument("--mode", default="diagnostic", choices=["diagnostic", "train"])
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--loss", default="return", choices=["return", "sharpe"])
    parser.add_argument("--diagnostic-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-stocks", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--num-epochs", type=int, default=TrainConfig.num_epochs)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=TrainConfig.grad_clip_norm)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=TrainConfig.early_stopping_patience,
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = PathsConfig()
    default_data_config = DataConfig()
    data_config = DataConfig(
        csv_path=args.data_path or default_data_config.csv_path,
        num_stocks=args.num_stocks,
    )
    train_config = TrainConfig(
        mode=args.mode,
        device=args.device,
        diagnostic_steps=args.diagnostic_steps,
        seed=args.seed,
        loss_name=args.loss,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        early_stopping_patience=args.early_stopping_patience,
    )
    metrics = run_training(data_config, ModelConfig(), train_config, paths)
    print(metrics)


if __name__ == "__main__":
    main()

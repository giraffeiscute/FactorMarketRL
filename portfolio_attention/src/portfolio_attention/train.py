"""Training entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio_attention.config import (
        DataConfig,
        DiagnosticConfig,
        ModelConfig,
        PathsConfig,
        TrainConfig,
    )
    from portfolio_attention.dataset import PortfolioPanelDataset
    from portfolio_attention.evaluate import run_diagnostic_evaluation
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
    from .config import DataConfig, DiagnosticConfig, ModelConfig, PathsConfig, TrainConfig
    from .dataset import PortfolioPanelDataset
    from .evaluate import run_diagnostic_evaluation
    from .losses import build_loss
    from .model import PortfolioAttentionModel
    from .utils import append_log, ensure_output_dirs, resolve_device, save_json, set_seed


TERMINAL_SUMMARY_KEYS = [
    "portfolio_return",
    "cash_weight",
    "stopped_early",
    "best_epoch",
    "best_val_loss",
    "source_path",
]
TERMINAL_METADATA_KEYS = [
    "total_num_days",
    "train_days",
    "dynamic_backtest_lookback_length",
    "dynamic_train_lookback_length",
    "dynamic_validation_lookback_length",
    "validation_days",
    "backtest_days",
    "analysis_horizon_days",
]
TERMINAL_OUTPUT_ORDER = TERMINAL_SUMMARY_KEYS + TERMINAL_METADATA_KEYS

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


def _resolve_model_config(model_config: ModelConfig, dataset: PortfolioPanelDataset) -> ModelConfig:
    return replace(model_config, lookback=dataset.model_lookback)


def _build_terminal_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    final_backtest = payload.get("final_backtest")
    if isinstance(final_backtest, dict):
        for key in TERMINAL_SUMMARY_KEYS:
            if key in final_backtest:
                summary[key] = final_backtest[key]
        metadata = final_backtest.get("metadata")
        if isinstance(metadata, dict):
            for key in TERMINAL_METADATA_KEYS:
                if key in metadata:
                    summary[key] = metadata[key]
    else:
        for key in TERMINAL_SUMMARY_KEYS:
            if key in payload:
                summary[key] = payload[key]
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for key in TERMINAL_METADATA_KEYS:
                if key in metadata:
                    summary[key] = metadata[key]
    for key in ("stopped_early", "best_epoch", "best_val_loss"):
        if key in payload:
            summary[key] = payload[key]
    return summary


def _format_terminal_summary(payload: dict[str, Any]) -> str:
    summary = _build_terminal_summary(payload)
    lines = [
        f"{key}: {summary[key]}"
        for key in TERMINAL_OUTPUT_ORDER
        if key in summary
    ]
    return "\n".join(lines)


def _append_dataset_split_summary(log_path: Path, dataset: PortfolioPanelDataset) -> None:
    metadata = dataset.metadata
    append_log(
        log_path,
        (
            "Dataset split summary: "
            f"total_num_days={metadata.total_num_days} "
            f"train_split_ratio={metadata.train_split_ratio:.4f} "
            f"validation_split_ratio={metadata.validation_split_ratio:.4f} "
            f"backtest_split_ratio={metadata.backtest_split_ratio:.4f} "
            f"train_split_length={metadata.train_split_length} "
            f"validation_split_length={metadata.validation_split_length} "
            f"backtest_split_length={metadata.backtest_split_length} "
            f"analysis_horizon_days={metadata.analysis_horizon_days} "
            f"train_horizon_start_index={metadata.train_horizon_start_index} "
            f"train_horizon_end_index={metadata.train_horizon_end_index} "
            f"validation_horizon_start_index={metadata.validation_horizon_start_index} "
            f"validation_horizon_end_index={metadata.validation_horizon_end_index} "
            f"backtest_horizon_start_index={metadata.backtest_horizon_start_index} "
            f"backtest_horizon_end_index={metadata.backtest_horizon_end_index} "
            f"dynamic_train_lookback_length={metadata.dynamic_train_lookback_length} "
            f"dynamic_validation_lookback_length={metadata.dynamic_validation_lookback_length} "
            f"dynamic_backtest_lookback_length={metadata.dynamic_backtest_lookback_length} "
            f"train_window_count={metadata.train_window_count} "
            f"validation_window_count={metadata.validation_window_count} "
            f"backtest_window_count={metadata.backtest_window_count}"
        ),
    )


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

    # Path-based interpretation:
    # Treat the current batch as a single continuous time-series path [1, T].
    # portfolio_return has shape [Batch], we unsqueeze to [1, Batch].
    path_returns = portfolio_return.unsqueeze(0)
    loss = build_loss(loss_name, path_returns)

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
        raise RuntimeError("Evaluation loader produced no samples.")

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
    resolved_model_config = _resolve_model_config(model_config, dataset)
    model = PortfolioAttentionModel(resolved_model_config, num_stocks=dataset.num_stocks).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    batch = dataset.get_backtest_batch(device=device)

    metrics: dict[str, float | int | bool | str | dict] = {
        "mode": "diagnostic",
        "device": str(device),
        "diagnostic_only": True,
        "evaluation_split": "backtest",
        "loaded_feature_columns": {
            "stock": dataset.loaded_stock_feature_columns,
            "market": dataset.loaded_market_feature_columns,
        },
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
    _append_dataset_split_summary(log_path, dataset)
    append_log(
        log_path,
        "Running diagnostic mode on the single fixed backtest sample defined by the backtest-split tail horizon.",
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
    train_dataset, validation_dataset, backtest_dataset = dataset.build_train_validation_backtest_datasets()
    if len(train_dataset) == 0 or len(validation_dataset) == 0 or len(backtest_dataset) == 0:
        raise RuntimeError(
            "Train mode requires exactly one train sample, one validation sample, and one backtest sample."
        )
    if train_config.epoch_print_interval <= 0:
        raise ValueError("TrainConfig.epoch_print_interval must be positive.")

    resolved_model_config = _resolve_model_config(model_config, dataset)
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
        shuffle=False,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
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
    _append_dataset_split_summary(log_path, dataset)
    append_log(
        log_path,
        (
            "Running epoch-based train mode with "
            f"train_windows={len(train_dataset)} validation_windows={len(validation_dataset)} "
            f"backtest_windows={len(backtest_dataset)} "
            f"batch_size={train_config.batch_size} num_epochs={train_config.num_epochs}."
        ),
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_completed = 0
    best_checkpoint_path = paths.checkpoints_dir / train_config.train_best_checkpoint_name
    last_checkpoint_path = paths.checkpoints_dir / train_config.train_last_checkpoint_name

    with tqdm(
        range(1, train_config.num_epochs + 1),
        total=train_config.num_epochs,
        desc=f"Epoch 0/{train_config.num_epochs}",
    ) as epoch_bar:
        for epoch in epoch_bar:
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
            val_loss, val_return = _evaluate_epoch(model, validation_loader, device, train_config.loss_name)

            epoch_metrics: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_portfolio_return": train_return,
                "val_loss": val_loss,
                "val_portfolio_return": val_return,
            }
            epochs_completed = epoch
            best_checkpoint_updated = False

            epoch_bar.set_description(f"Epoch {epoch}/{train_config.num_epochs}")
            epoch_bar.set_postfix(
                train_loss=f"{epoch_metrics['train_loss']:.6f}",
                train_ret=f"{epoch_metrics['train_portfolio_return']:.6f}",
                val_loss=f"{epoch_metrics['val_loss']:.6f}",
                val_ret=f"{epoch_metrics['val_portfolio_return']:.6f}",
            )

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
                best_checkpoint_updated = True
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
                        extra_metrics={
                            **epoch_metrics,
                            "best_checkpoint_updated": best_checkpoint_updated,
                            "epochs_without_improvement": epochs_without_improvement,
                        },
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
                    extra_metrics={
                        **epoch_metrics,
                        "best_checkpoint_updated": best_checkpoint_updated,
                        "epochs_without_improvement": epochs_without_improvement,
                    },
                ),
                last_checkpoint_path,
            )

            should_stop_early = epochs_without_improvement >= train_config.early_stopping_patience
            if epoch % train_config.epoch_print_interval == 0 or epoch == train_config.num_epochs or should_stop_early:
                tqdm.write(
                    (
                        f"[Epoch {epoch}/{train_config.num_epochs}] "
                        f"train_loss={epoch_metrics['train_loss']:.6f} "
                        f"train_portfolio_return={epoch_metrics['train_portfolio_return']:.6f} "
                        f"val_loss={epoch_metrics['val_loss']:.6f} "
                        f"val_portfolio_return={epoch_metrics['val_portfolio_return']:.6f}"
                    )
                )

            if should_stop_early:
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

    append_log(log_path, f"Loading best checkpoint for final backtest evaluation: {best_checkpoint_path}.")
    final_backtest = run_diagnostic_evaluation(
        data_config=data_config,
        paths=paths,
        checkpoint_path=best_checkpoint_path,
        device_name=train_config.device,
        diagnostic_config=DiagnosticConfig(),
        diagnostic_only=False,
    )

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
        "train_window_count": len(train_dataset),
        "validation_window_count": len(validation_dataset),
        "backtest_window_count": len(backtest_dataset),
        "early_stopping_patience": train_config.early_stopping_patience,
        "stopped_early": epochs_completed < train_config.num_epochs,
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "final_backtest": final_backtest,
        "metadata": dataset.metadata.as_dict(),
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
    parser.add_argument("--mode", default=argparse.SUPPRESS, choices=["diagnostic", "train"])
    parser.add_argument("--data-path", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument(
        "--loss",
        default=argparse.SUPPRESS,
        choices=[
            "return",
            "terminal_return",
            "sharpe",
            "dsr",
            "sortino",
            "mdd",
            "cvar",
        ],
    )
    parser.add_argument("--diagnostic-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-stocks", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--epoch-log-interval", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--grad-clip-norm", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=argparse.SUPPRESS,
    )
    return parser


def resolve_runtime_configs_from_args(
    args: argparse.Namespace,
    *,
    data_config: DataConfig | None = None,
    train_config: TrainConfig | None = None,
) -> tuple[DataConfig, TrainConfig]:
    resolved_data_config = data_config or DataConfig()
    resolved_train_config = train_config or TrainConfig()
    args_dict = vars(args)

    data_overrides = {}
    if "data_path" in args_dict:
        data_overrides["csv_path"] = args_dict["data_path"]
    if "num_stocks" in args_dict:
        data_overrides["num_stocks"] = args_dict["num_stocks"]
    if data_overrides:
        resolved_data_config = replace(resolved_data_config, **data_overrides)

    train_overrides = {}
    if "mode" in args_dict:
        train_overrides["mode"] = args_dict["mode"]
    if "device" in args_dict:
        train_overrides["device"] = args_dict["device"]
    if "diagnostic_steps" in args_dict:
        train_overrides["diagnostic_steps"] = args_dict["diagnostic_steps"]
    if "seed" in args_dict:
        train_overrides["seed"] = args_dict["seed"]
    if "loss" in args_dict:
        train_overrides["loss_name"] = args_dict["loss"]
    if "batch_size" in args_dict:
        train_overrides["batch_size"] = args_dict["batch_size"]
    if "num_epochs" in args_dict:
        train_overrides["num_epochs"] = args_dict["num_epochs"]
    if "epoch_print_interval" in args_dict:
        train_overrides["epoch_print_interval"] = args_dict["epoch_print_interval"]
    if "weight_decay" in args_dict:
        train_overrides["weight_decay"] = args_dict["weight_decay"]
    if "grad_clip_norm" in args_dict:
        train_overrides["grad_clip_norm"] = args_dict["grad_clip_norm"]
    if "early_stopping_patience" in args_dict:
        train_overrides["early_stopping_patience"] = args_dict["early_stopping_patience"]
    if train_overrides:
        resolved_train_config = replace(resolved_train_config, **train_overrides)

    return resolved_data_config, resolved_train_config


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = PathsConfig()
    data_config, train_config = resolve_runtime_configs_from_args(args)
    metrics = run_training(data_config, ModelConfig(), train_config, paths)
    print(_format_terminal_summary(metrics))


if __name__ == "__main__":
    main()

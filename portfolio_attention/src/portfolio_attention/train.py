"""Training entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

try:
    from rich.live import Live
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio_attention.config import (
        DataConfig,
        ModelConfig,
        PathsConfig,
        TrainConfig,
    )
    from portfolio_attention.dataset import PortfolioPanelDataset
    from portfolio_attention.evaluate import run_evaluation
    from portfolio_attention.losses import build_loss
    from portfolio_attention.model import PortfolioAttentionModel
    from portfolio_attention.utils import (
        append_log,
        ensure_output_dirs,
        format_determinism_status,
        get_determinism_status,
        resolve_device,
        save_json,
        set_seed,
    )
else:
    from .config import DataConfig, ModelConfig, PathsConfig, TrainConfig
    from .dataset import PortfolioPanelDataset
    from .evaluate import run_evaluation
    from .losses import build_loss
    from .model import PortfolioAttentionModel
    from .utils import (
        append_log,
        ensure_output_dirs,
        format_determinism_status,
        get_determinism_status,
        resolve_device,
        save_json,
        set_seed,
    )


TERMINAL_SUMMARY_KEYS = [
    "seed",
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


def _write_training_status(
    paths: PathsConfig,
    loss_name: str,
    status: str,
    **kwargs,
) -> None:
    """Writes current training status to a JSON file."""
    status_path = _status_path_for_loss(paths, loss_name)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "loss_name": loss_name,
        "status": status,
        "pid": os.getpid(),
        "updated_at": time.time(),
        **kwargs,
    }
    # Persistence of started_at
    if status == "RUNNING" and "started_at" not in payload:
        try:
            with open(status_path, "r") as f:
                old_payload = json.load(f)
                if "started_at" in old_payload:
                    payload["started_at"] = old_payload["started_at"]
                else:
                    payload["started_at"] = time.time()
        except (FileNotFoundError, json.JSONDecodeError):
            payload["started_at"] = time.time()
    elif "started_at" not in payload and status in ("DONE", "FAILED"):
        try:
             with open(status_path, "r") as f:
                old_payload = json.load(f)
                if "started_at" in old_payload:
                    payload["started_at"] = old_payload["started_at"]
        except (FileNotFoundError, json.JSONDecodeError):
             pass

    with open(status_path, "w") as f:
        json.dump(payload, f, indent=2)


def _load_training_status(paths: PathsConfig, loss_name: str) -> dict[str, Any]:
    """Loads training status from a JSON file."""
    status_path = _status_path_for_loss(paths, loss_name)
    if not status_path.exists():
        return {"loss_name": loss_name, "status": "QUEUED"}
    try:
        with open(status_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"loss_name": loss_name, "status": "UNKNOWN"}


def _status_path_for_loss(paths: PathsConfig, loss_name: str) -> Path:
    return paths.status_dir / f"train_status_{loss_name}.json"


def _read_metrics_for_loss(paths: PathsConfig, loss_name: str) -> dict[str, Any] | None:
    """Reads final metrics JSON for a loss."""
    metrics_path = paths.metrics_dir / f"train_metrics_{loss_name}.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _render_multi_loss_dashboard(
    paths: PathsConfig,
    losses: list[str],
    active_jobs: dict[str, dict[str, Any]],
) -> Any:
    """Renders a dashboard table using rich."""
    if not HAS_RICH:
        # Simple fallback
        lines = ["\n" + "=" * 50]
        for loss in losses:
            status_data = _load_training_status(paths, loss)
            status = status_data.get("status", "QUEUED")
            epoch = status_data.get("epoch", 0)
            total = status_data.get("num_epochs", "?")
            train_loss = status_data.get("train_loss", 0.0)
            val_loss = status_data.get("val_loss", 0.0)
            lines.append(f"{loss:<10} | {status:<8} | Epoch: {epoch}/{total} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        return "\n".join(lines)

    table = Table(title="Multi-Loss Portfolio Training Dashboard", show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Loss Name", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Device/GPU", style="green")
    table.add_column("Progress", style="white")
    table.add_column("Epoch", justify="right")
    table.add_column("Train Loss", justify="right")
    table.add_column("Val Loss", justify="right")
    table.add_column("Best Val", justify="right")

    for loss in losses:
        data = _load_training_status(paths, loss)
        status = data.get("status", "QUEUED")
        
        # Color coding for status
        status_display = status
        if status == "RUNNING":
            status_display = "[bold blue]RUNNING[/bold blue]"
        elif status == "DONE":
            status_display = "[bold green]DONE[/bold green]"
        elif status == "FAILED":
            status_display = "[bold red]FAILED[/bold red]"
        elif status == "QUEUED":
            status_display = "[dim]QUEUED[/dim]"

        device = data.get("device", "-")
        epoch = data.get("epoch", 0)
        total_epochs = data.get("num_epochs", 0)
        progress_ratio = data.get("progress_ratio", 0.0)
        
        # Progress bar string
        bar_len = 10
        filled = int(progress_ratio * bar_len)
        prog_bar = "[" + "=" * filled + " " * (bar_len - filled) + "]"
        prog_display = f"{prog_bar} {progress_ratio*100:>3.0f}%"

        train_loss = data.get("train_loss", 0.0)
        val_loss = data.get("val_loss", 0.0)
        best_val = data.get("best_val_loss", 0.0)

        table.add_row(
            loss,
            status_display,
            str(device),
            prog_display,
            f"{epoch}/{total_epochs}",
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{best_val:.6f}"
        )

    return table


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _log_reproducibility_status(log_path: Path, train_config: TrainConfig, device: torch.device) -> None:
    status = get_determinism_status(device=device, seed=train_config.seed)
    message = format_determinism_status(status)
    print(message, flush=True)
    append_log(log_path, message)


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
    for key in ("seed", "stopped_early", "best_epoch", "best_val_loss"):
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
    model_config: ModelConfig,
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
        "model_config": model_config.as_dict(),
        "max_lookback": model.max_lookback,
        "data_config": _serialize_config(data_config),
        "train_config": _serialize_config(train_config),
        "metadata": dataset.metadata.as_dict(),
    }
    if extra_metrics:
        payload["metrics"] = extra_metrics
    return payload


def _normalize_best_epoch_selection_window(select_best_from_last_x_epochs: int) -> int:
    """Normalizes the trailing best-epoch selection window.

    The best epoch is selected only from the last X completed epochs.
    X=1 means only the final completed epoch is eligible. When X is larger
    than the number of completed epochs, all completed epochs remain eligible.
    Non-positive values are normalized to 1 to avoid falling back to a
    global-best-over-all-epochs selection rule.
    """
    return max(1, int(select_best_from_last_x_epochs))


def _epoch_candidate_checkpoint_path(paths: PathsConfig, loss_name: str, epoch: int) -> Path:
    return paths.checkpoints_dir / f"train_candidate_{loss_name}_epoch_{epoch}.pt"


def _select_best_epoch_record(
    epoch_records: list[dict[str, Any]],
    select_best_from_last_x_epochs: int,
) -> dict[str, Any]:
    if not epoch_records:
        raise RuntimeError("No epoch records were collected for best-epoch selection.")

    normalized_window = _normalize_best_epoch_selection_window(select_best_from_last_x_epochs)
    candidate_records = epoch_records[-normalized_window:]
    return min(candidate_records, key=lambda record: (float(record["val_loss"]), int(record["epoch"])))


def _cleanup_temp_epoch_checkpoints(epoch_records: list[dict[str, Any]]) -> None:
    seen_paths: set[Path] = set()
    for record in epoch_records:
        checkpoint_path = record.get("checkpoint_path")
        if checkpoint_path is None:
            continue
        path = Path(str(checkpoint_path))
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


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


def run_epoch_training(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    paths: PathsConfig,
) -> dict:
    set_seed(train_config.seed)
    ensure_output_dirs(paths)
    device = resolve_device(train_config.device)
    log_path = paths.logs_dir / f"train_{train_config.loss_name}.log"
    _log_reproducibility_status(log_path, train_config, device)

    dataset = PortfolioPanelDataset(data_config)
    train_dataset, validation_dataset, backtest_dataset = dataset.build_train_validation_backtest_datasets()
    if len(train_dataset) == 0 or len(validation_dataset) == 0 or len(backtest_dataset) == 0:
        raise RuntimeError("Training requires exactly one train sample, one validation sample, and one backtest sample.")

    model = PortfolioAttentionModel(
        model_config,
        num_stocks=dataset.num_stocks,
        max_lookback=dataset.model_lookback,
    ).to(device)
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

    append_log(
        log_path,
        (
            "Loaded feature columns successfully: "
            f"stock={dataset.loaded_stock_feature_columns} "
            f"market={dataset.loaded_market_feature_columns}"
        ),
    )
    _append_dataset_split_summary(log_path, dataset)

    selection_window = _normalize_best_epoch_selection_window(
        train_config.select_best_from_last_x_epochs
    )
    append_log(
        log_path,
        (
            "Running epoch-based training with "
            f"train_windows={len(train_dataset)} validation_windows={len(validation_dataset)} "
            f"backtest_windows={len(backtest_dataset)} "
            f"batch_size={train_config.batch_size} num_epochs={train_config.num_epochs} "
            f"select_best_from_last_x_epochs={train_config.select_best_from_last_x_epochs} "
            f"normalized_best_epoch_selection_window={selection_window}."
        ),
    )

    global_best_val_loss = float("inf")
    epochs_without_improvement = 0
    epochs_completed = 0
    best_checkpoint_path = paths.checkpoints_dir / train_config.train_best_checkpoint_name
    last_checkpoint_path = paths.checkpoints_dir / train_config.train_last_checkpoint_name
    epoch_selection_records: list[dict[str, Any]] = []
    history: list[dict[str, float | int | bool]] = []

    # Initial status update
    _write_training_status(
        paths,
        train_config.loss_name,
        "RUNNING",
        device=str(device),
        epoch=0,
        num_epochs=train_config.num_epochs,
        progress_ratio=0.0,
    )

    try:
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
            val_loss, val_return = _evaluate_epoch(model, validation_loader, device, train_config.loss_name)

            epoch_metrics: dict[str, float | int | bool] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_portfolio_return": train_return,
                "val_loss": val_loss,
                "val_portfolio_return": val_return,
            }
            history.append(dict(epoch_metrics))
            epochs_completed = epoch
            global_best_checkpoint_updated = False

            append_log(
                log_path,
                (
                    f"epoch={epoch} train_loss={train_loss:.8f} "
                    f"train_portfolio_return={train_return:.8f} "
                    f"val_loss={val_loss:.8f} val_portfolio_return={val_return:.8f}"
                ),
            )

            if val_loss < global_best_val_loss:
                global_best_val_loss = val_loss
                epochs_without_improvement = 0
                global_best_checkpoint_updated = True
            else:
                epochs_without_improvement += 1

            candidate_checkpoint_path = _epoch_candidate_checkpoint_path(paths, train_config.loss_name, epoch)
            torch.save(
                _build_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    model_config=model_config,
                    data_config=data_config,
                    train_config=train_config,
                    dataset=dataset,
                    epoch=epoch,
                    best_val_loss=val_loss,
                    extra_metrics={
                        **epoch_metrics,
                        "global_best_checkpoint_updated": global_best_checkpoint_updated,
                        "epochs_without_improvement": epochs_without_improvement,
                    },
                ),
                candidate_checkpoint_path,
            )
            epoch_selection_records.append(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "checkpoint_path": str(candidate_checkpoint_path),
                }
            )

            if len(epoch_selection_records) > selection_window:
                stale_record = epoch_selection_records.pop(0)
                stale_checkpoint_path = Path(str(stale_record["checkpoint_path"]))
                try:
                    stale_checkpoint_path.unlink()
                except FileNotFoundError:
                    pass

            current_window_best_record = _select_best_epoch_record(
                epoch_selection_records,
                selection_window,
            )
            current_window_best_epoch = int(current_window_best_record["epoch"])
            current_window_best_val_loss = float(current_window_best_record["val_loss"])

            torch.save(
                _build_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    model_config=model_config,
                    data_config=data_config,
                    train_config=train_config,
                    dataset=dataset,
                    epoch=epoch,
                    best_val_loss=current_window_best_val_loss,
                    extra_metrics={
                        **epoch_metrics,
                        "current_window_best_epoch": current_window_best_epoch,
                        "current_window_best_val_loss": current_window_best_val_loss,
                        "global_best_val_loss": global_best_val_loss,
                        "global_best_checkpoint_updated": global_best_checkpoint_updated,
                        "epochs_without_improvement": epochs_without_improvement,
                    },
                ),
                last_checkpoint_path,
            )

            _write_training_status(
                paths,
                train_config.loss_name,
                "RUNNING",
                device=str(device),
                epoch=epoch,
                num_epochs=train_config.num_epochs,
                progress_ratio=epoch / train_config.num_epochs,
                train_loss=train_loss,
                train_portfolio_return=train_return,
                val_loss=val_loss,
                val_portfolio_return=val_return,
                best_epoch=current_window_best_epoch,
                best_val_loss=current_window_best_val_loss,
                global_best_val_loss=global_best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
                select_best_from_last_x_epochs=selection_window,
            )

            should_stop_early = epochs_without_improvement >= train_config.early_stopping_patience
            if should_stop_early:
                append_log(
                    log_path,
                    (
                        "Early stopping triggered with "
                        f"patience={train_config.early_stopping_patience} at epoch={epoch}."
                    ),
                )
                break
    except Exception as e:
        _write_training_status(
            paths,
            train_config.loss_name,
            "FAILED",
            error_message=str(e),
        )
        raise e

    if not epoch_selection_records:
        raise RuntimeError("Train loop did not record any epoch candidates for best selection.")

    selected_best_record = _select_best_epoch_record(
        epoch_selection_records,
        selection_window,
    )
    best_epoch = int(selected_best_record["epoch"])
    best_val_loss = float(selected_best_record["val_loss"])
    selected_best_checkpoint_path = Path(str(selected_best_record["checkpoint_path"]))
    effective_selection_window = min(selection_window, epochs_completed)

    append_log(
        log_path,
        (
            "Selecting final best checkpoint from trailing validation window: "
            f"configured_window={train_config.select_best_from_last_x_epochs} "
            f"normalized_window={selection_window} "
            f"effective_window={effective_selection_window} "
            f"selected_best_epoch={best_epoch} "
            f"selected_best_val_loss={best_val_loss:.8f}."
        ),
    )

    shutil.copy2(selected_best_checkpoint_path, best_checkpoint_path)
    _cleanup_temp_epoch_checkpoints(epoch_selection_records)

    append_log(log_path, f"Loading best checkpoint for final backtest evaluation: {best_checkpoint_path}.")
    final_backtest = run_evaluation(
        data_config=data_config,
        paths=paths,
        checkpoint_path=best_checkpoint_path,
        device_name=train_config.device,
    )

    metrics: dict[str, Any] = {
        "mode": "train",
        "device": str(device),
        "loaded_feature_columns": {
            "stock": dataset.loaded_stock_feature_columns,
            "market": dataset.loaded_market_feature_columns,
        },
        "loss_name": train_config.loss_name,
        "seed": train_config.seed,
        "batch_size": train_config.batch_size,
        "num_epochs_requested": train_config.num_epochs,
        "epochs_completed": epochs_completed,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "select_best_from_last_x_epochs": train_config.select_best_from_last_x_epochs,
        "normalized_best_epoch_selection_window": selection_window,
        "effective_best_epoch_selection_window": effective_selection_window,
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
    save_json(metrics, paths.metrics_dir / f"train_metrics_{train_config.loss_name}.json")

    _write_training_status(
        paths,
        train_config.loss_name,
        "DONE",
        device=str(device),
        epoch=epochs_completed,
        num_epochs=train_config.num_epochs,
        progress_ratio=1.0,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        stopped_early=epochs_completed < train_config.num_epochs,
        select_best_from_last_x_epochs=selection_window,
    )

    return metrics


def run_training(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    paths: PathsConfig,
) -> dict:
    return run_epoch_training(data_config, model_config, train_config, paths)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run training for portfolio_attention.")
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
    parser.add_argument("--losses", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-stocks", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--grad-clip-norm", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--select-best-from-last-x-epochs",
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
    if "device" in args_dict:
        train_overrides["device"] = args_dict["device"]
    if "seed" in args_dict:
        train_overrides["seed"] = args_dict["seed"]
    if "loss" in args_dict:
        train_overrides["loss_name"] = args_dict["loss"]
    if "batch_size" in args_dict:
        train_overrides["batch_size"] = args_dict["batch_size"]
    if "num_epochs" in args_dict:
        train_overrides["num_epochs"] = args_dict["num_epochs"]
    if "weight_decay" in args_dict:
        train_overrides["weight_decay"] = args_dict["weight_decay"]

    if "grad_clip_norm" in args_dict:
        train_overrides["grad_clip_norm"] = args_dict["grad_clip_norm"]
    if "early_stopping_patience" in args_dict:
        train_overrides["early_stopping_patience"] = args_dict["early_stopping_patience"]
    if "select_best_from_last_x_epochs" in args_dict:
        train_overrides["select_best_from_last_x_epochs"] = args_dict["select_best_from_last_x_epochs"]
    if train_overrides:
        resolved_train_config = replace(resolved_train_config, **train_overrides)

    return resolved_data_config, resolved_train_config


DEFAULT_LOSSES = ["return", "sharpe", "dsr", "sortino", "mdd", "cvar"]


def _normalize_losses(raw_losses: list[str]) -> list[str]:
    valid_losses = {"return", "sharpe", "dsr", "sortino", "mdd", "cvar"}
    result = []
    seen = set()
    for loss in raw_losses:
        loss = loss.strip()
        if not loss:
            continue
        if loss == "terminal_return":
            loss = "return"
        if loss not in valid_losses:
            raise ValueError(f"Invalid loss: '{loss}'. Must be one of {valid_losses} or 'terminal_return'")
        if loss not in seen:
            seen.add(loss)
            result.append(loss)
    return result


def _parse_losses_args(args: argparse.Namespace) -> list[str]:
    args_dict = vars(args)
    if "loss" in args_dict:
        return _normalize_losses([args_dict["loss"]])
    if "losses" in args_dict:
        val = args_dict["losses"]
        if not val or not val.strip():
            raise ValueError("--losses cannot be empty string")
        return _normalize_losses(val.split(","))
    return list(DEFAULT_LOSSES)


def _resolve_round_robin_gpu_ids(parallel: int) -> list[int]:
    if parallel <= 0:
        raise ValueError("parallel must be positive")
    if not torch.cuda.is_available():
        return []
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        return []
    return list(range(min(4, gpu_count)))


def _build_subprocess_cmd(loss: str, device: str | None = None) -> list[str]:
    cmd = [sys.executable, os.path.abspath(__file__)]
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("--losses=") or arg.startswith("--parallel="):
            continue
        if arg in ("--losses", "--parallel"):
            skip_next = True
            continue
        if arg.startswith("--loss="):
            continue
        if arg == "--loss":
            skip_next = True
            continue
        if arg.startswith("--device="):
            continue
        if arg == "--device":
            skip_next = True
            continue
        cmd.append(arg)
    cmd.extend(["--loss", loss])
    if device is not None:
        cmd.extend(["--device", device])
    return cmd


def _is_worker_mode() -> bool:
    return os.environ.get("PORTFOLIO_ATTENTION_CHILD") == "1"


def main() -> None:
    args = build_arg_parser().parse_args()
    args_dict = vars(args)
    paths = PathsConfig()

    parallel = args_dict.get("parallel", 1)
    if parallel < 1:
        raise ValueError("--parallel must be >= 1")

    losses_to_run = _parse_losses_args(args)
    worker_mode = _is_worker_mode()

    if "loss" in args_dict:
        # Run single loss directly in the current process
        loss = losses_to_run[0]
        args.loss = loss
        data_config, train_config = resolve_runtime_configs_from_args(args)
        
        if not worker_mode:
            print(f"\n>>> Running training with loss: {loss}")
        
        try:
            metrics = run_training(data_config, ModelConfig(), train_config, paths)
            if not worker_mode:
                print(f"--- Results for loss: {loss} ---")
                print(_format_terminal_summary(metrics))
        except Exception:
            if worker_mode:
                # Child process should re-raise to exit with non-zero
                raise
            else:
                print(f"ERROR: Training for loss '{loss}' failed.")
                sys.exit(1)
        return

    # Multi loss mode (spawn subprocesses)
    gpu_ids = _resolve_round_robin_gpu_ids(parallel)
    
    # Clean up old status files before starting
    if paths.status_dir.exists():
        for f in paths.status_dir.glob("train_status_*.json"):
            try:
                f.unlink()
            except OSError:
                pass

    active_processes: list[tuple[str, int | None, subprocess.Popen]] = []
    pending_losses = list(losses_to_run)
    failed_losses: list[str] = []
    launch_index = 0

    dashboard_context = Live(_render_multi_loss_dashboard(paths, losses_to_run, {}), refresh_per_second=2) if HAS_RICH else None
    if dashboard_context:
        dashboard_context.start()

    try:
        while pending_losses or active_processes:
            # Start new processes if we have capacity
            while pending_losses and len(active_processes) < parallel:
                loss = pending_losses.pop(0)
                child_env = os.environ.copy()
                child_env["PORTFOLIO_ATTENTION_CHILD"] = "1"
                
                stdout_path = paths.logs_dir / f"train_{loss}.stdout.log"
                stderr_path = paths.logs_dir / f"train_{loss}.stderr.log"
                stdout_file = open(stdout_path, "w")
                stderr_file = open(stderr_path, "w")

                if gpu_ids:
                    physical_gpu_id = gpu_ids[launch_index % len(gpu_ids)]
                    child_env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
                    cmd = _build_subprocess_cmd(loss, device="cuda:0")
                    p = subprocess.Popen(cmd, env=child_env, stdout=stdout_file, stderr=stderr_file)
                else:
                    physical_gpu_id = None
                    cmd = _build_subprocess_cmd(loss, device="cpu")
                    p = subprocess.Popen(cmd, env=child_env, stdout=stdout_file, stderr=stderr_file)

                # Store file handles to close later
                active_processes.append((loss, physical_gpu_id, p, stdout_file, stderr_file))
                launch_index += 1

            # Check active processes
            still_active = []
            for loss, physical_gpu_id, p, out_f, err_f in active_processes:
                retcode = p.poll()
                if retcode is None:
                    still_active.append((loss, physical_gpu_id, p, out_f, err_f))
                else:
                    out_f.close()
                    err_f.close()
                    if retcode != 0:
                        failed_losses.append(loss)
            
            active_processes = still_active
            
            # Update Dashboard
            if dashboard_context:
                dashboard_context.update(_render_multi_loss_dashboard(paths, losses_to_run, {}))
            elif not HAS_RICH:
                # Simple fallback printing
                print(_render_multi_loss_dashboard(paths, losses_to_run, {}), end="\r")

            if active_processes:
                time.sleep(0.5)
    finally:
        if dashboard_context:
            dashboard_context.stop()

    # Final Summaries
    print("\n" + "=" * 50)
    print("FINAL TRAINING SUMMARIES")
    print("=" * 50)
    
    for loss in losses_to_run:
        metrics = _read_metrics_for_loss(paths, loss)
        if metrics:
            print(f"\n--- Results for loss: {loss} ---")
            print(_format_terminal_summary(metrics))
        else:
            status_data = _load_training_status(paths, loss)
            err_msg = status_data.get("error_message", "Unknown error")
            print(f"\n--- Results for loss: {loss} ---")
            print(f"STATUS: FAILED")
            print(f"ERROR: {err_msg}")
            print(f"Check logs: {paths.logs_dir}/train_{loss}.stderr.log")

    if failed_losses:
        print(f"\nTraining completed with failures in: {', '.join(failed_losses)}")
        sys.exit(1)
    else:
        print("\nAll training processes completed successfully.")


if __name__ == "__main__":
    main()

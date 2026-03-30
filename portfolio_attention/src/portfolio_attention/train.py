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
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio_attention.config import DataConfig, ModelConfig, PathsConfig, TrainConfig
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
    "state",
    "loss_name",
    "mean_final_return",
    "std_final_return",
    "median_final_return",
    "worst_scenario_final_return",
    "best_scenario_final_return",
    "best_scenario_id",
    "best_epoch",
    "best_val_loss",
]

NON_TERMINAL_STATUSES = {"QUEUED", "STARTING", "PREPARING_DATA", "RUNNING"}


def _serialize_config(config: object) -> dict[str, Any]:
    serialized = asdict(config)  # type: ignore[arg-type]
    for key, value in list(serialized.items()):
        if isinstance(value, Path):
            serialized[key] = str(value)
    return serialized


def _status_path_for_loss(paths: PathsConfig, loss_name: str) -> Path:
    return paths.status_dir / f"train_status_{loss_name}.json"


def _console_log_path_for_loss(paths: PathsConfig, loss_name: str) -> Path:
    return paths.logs_dir / f"train_{loss_name}.console.log"


def _write_training_status(
    paths: PathsConfig,
    loss_name: str,
    status: str,
    **kwargs: Any,
) -> None:
    status_path = _status_path_for_loss(paths, loss_name)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        previous_payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        previous_payload = {}

    payload = {
        "loss_name": loss_name,
        "status": status,
        "pid": os.getpid(),
        "updated_at": time.time(),
        "phase": kwargs.get("phase", previous_payload.get("phase", "queued")),
        "message": kwargs.get("message", previous_payload.get("message", "")),
        **kwargs,
    }
    if "started_at" not in payload:
        if status in NON_TERMINAL_STATUSES:
            payload["started_at"] = previous_payload.get("started_at", time.time())
        elif status in {"DONE", "FAILED"} and "started_at" in previous_payload:
            payload["started_at"] = previous_payload["started_at"]

    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_training_status(paths: PathsConfig, loss_name: str) -> dict[str, Any]:
    status_path = _status_path_for_loss(paths, loss_name)
    if not status_path.exists():
        return {
            "loss_name": loss_name,
            "status": "QUEUED",
            "phase": "queued",
            "message": "Waiting to start.",
        }
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "loss_name": loss_name,
            "status": "UNKNOWN",
            "phase": "status_error",
            "message": "Could not parse status file.",
        }
    payload.setdefault("loss_name", loss_name)
    payload.setdefault("status", "UNKNOWN")
    payload.setdefault("phase", "unknown")
    payload.setdefault("message", "")
    return payload


def _status_snapshot(paths: PathsConfig, losses: list[str]) -> list[dict[str, Any]]:
    return [_load_training_status(paths, loss) for loss in losses]


def _dashboard_signature(status_rows: list[dict[str, Any]]) -> str:
    projected_rows = [
        {
            "loss_name": row.get("loss_name"),
            "status": row.get("status"),
            "phase": row.get("phase"),
            "device": row.get("device"),
            "epoch": row.get("epoch"),
            "num_epochs": row.get("num_epochs"),
            "train_loss": row.get("train_loss"),
            "val_loss": row.get("val_loss"),
            "best_val_loss": row.get("best_val_loss"),
            "message": row.get("message"),
        }
        for row in status_rows
    ]
    return json.dumps(projected_rows, sort_keys=True, default=str)


def _should_use_live_dashboard() -> bool:
    if not HAS_RICH:
        return False
    is_tty = getattr(sys.stdout, "isatty", lambda: False)()
    term = os.environ.get("TERM", "")
    return bool(is_tty and term and term.lower() != "dumb")


def _render_multi_loss_dashboard(
    status_rows: list[dict[str, Any]],
) -> Any:
    if not _should_use_live_dashboard():
        lines = ["Multi-loss training status"]
        for status_data in status_rows:
            loss = str(status_data.get("loss_name", "unknown"))
            epoch = int(status_data.get("epoch", 0))
            raw_num_epochs = status_data.get("num_epochs", "?")
            num_epochs = raw_num_epochs if raw_num_epochs not in (None, 0) else "?"
            lines.append(
                f"{loss:<10} | {str(status_data.get('status', 'QUEUED')):<14} | "
                f"{str(status_data.get('phase', 'queued')):<16} | "
                f"Epoch {epoch}/{num_epochs} | "
                f"Train {float(status_data.get('train_loss', 0.0)):.6f} | "
                f"Val {float(status_data.get('val_loss', 0.0)):.6f} | "
                f"Best {float(status_data.get('best_val_loss', 0.0)):.6f}"
            )
        return "\n".join(lines)

    table = Table(
        title="Multi-Loss Portfolio Training Dashboard",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("Loss", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Phase", style="white")
    table.add_column("Device", style="green")
    table.add_column("Epoch", justify="right")
    table.add_column("Train Loss", justify="right")
    table.add_column("Val Loss", justify="right")
    table.add_column("Best Val", justify="right")

    for data in status_rows:
        loss = str(data.get("loss_name", "unknown"))
        status = data.get("status", "QUEUED")
        status_display = status
        if status == "RUNNING":
            status_display = "[bold blue]RUNNING[/bold blue]"
        elif status == "DONE":
            status_display = "[bold green]DONE[/bold green]"
        elif status == "FAILED":
            status_display = "[bold red]FAILED[/bold red]"
        elif status == "STARTING":
            status_display = "[yellow]STARTING[/yellow]"
        elif status == "PREPARING_DATA":
            status_display = "[yellow]PREPARING[/yellow]"
        elif status == "QUEUED":
            status_display = "[dim]QUEUED[/dim]"
        raw_num_epochs = data.get("num_epochs", 0)
        num_epochs = raw_num_epochs if raw_num_epochs not in (None, 0) else "?"

        table.add_row(
            loss,
            status_display,
            str(data.get("phase", "-")),
            str(data.get("device", "-")),
            f"{data.get('epoch', 0)}/{num_epochs}",
            f"{float(data.get('train_loss', 0.0)):.6f}",
            f"{float(data.get('val_loss', 0.0)):.6f}",
            f"{float(data.get('best_val_loss', 0.0)):.6f}",
        )
    return table


def _tail_console_log(paths: PathsConfig, loss_name: str, *, num_lines: int = 8) -> str:
    console_log_path = _console_log_path_for_loss(paths, loss_name)
    try:
        lines = console_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    tail = lines[-num_lines:]
    return "\n".join(tail)


def _build_failure_summary(paths: PathsConfig, loss_name: str, returncode: int | None) -> str:
    status_data = _load_training_status(paths, loss_name)
    summary_lines = [
        f"Loss '{loss_name}' failed with exit code {returncode}.",
    ]
    error_message = str(status_data.get("error_message", "")).strip()
    if error_message:
        summary_lines.append(f"Status error: {error_message}")
    log_tail = _tail_console_log(paths, loss_name)
    if log_tail:
        summary_lines.append("Console log tail:")
        summary_lines.append(log_tail)
    return "\n".join(summary_lines)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _log_reproducibility_status(log_path: Path, train_config: TrainConfig, device: torch.device) -> None:
    status = get_determinism_status(device=device, seed=train_config.seed)
    message = format_determinism_status(status)
    if not _is_worker_mode():
        print(message, flush=True)
    append_log(log_path, message)


def _build_terminal_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    final_backtest = payload.get("final_backtest")
    if isinstance(final_backtest, dict):
        for key in TERMINAL_SUMMARY_KEYS:
            if key in final_backtest:
                summary[key] = final_backtest[key]
    for key in ("best_epoch", "best_val_loss"):
        if key in payload:
            summary[key] = payload[key]
    if "loss_name" in payload:
        summary.setdefault("loss_name", payload["loss_name"])
    return summary


def _format_terminal_summary(payload: dict[str, Any]) -> str:
    summary = _build_terminal_summary(payload)
    return "\n".join(
        f"{key}: {summary[key]}"
        for key in TERMINAL_SUMMARY_KEYS + ["best_epoch", "best_val_loss"]
        if key in summary
    )


def _append_dataset_split_summary(log_path: Path, dataset: PortfolioPanelDataset) -> None:
    metadata = dataset.metadata
    append_log(
        log_path,
        (
            f"Found {metadata.total_scenarios_found} scenarios in {metadata.scenario_dir} "
            f"using glob='{metadata.scenario_glob}'."
        ),
    )
    append_log(log_path, f"Train scenarios ({metadata.num_train_scenarios}): {metadata.train_scenarios}")
    append_log(
        log_path,
        f"Validation scenarios ({metadata.num_validation_scenarios}): {metadata.validation_scenarios}",
    )
    append_log(log_path, f"Holdout test scenarios ({metadata.num_test_scenarios}): {metadata.test_scenarios}")
    append_log(
        log_path,
        (
            "Scenario time split lengths: "
            f"train_raw={metadata.train_segment_raw_length} train_time_steps={metadata.train_segment_time_steps} | "
            f"validation_raw={metadata.validation_segment_raw_length} "
            f"validation_time_steps={metadata.validation_segment_time_steps} | "
            f"test_raw={metadata.test_segment_raw_length} test_time_steps={metadata.test_segment_time_steps}"
        ),
    )
    append_log(
        log_path,
        (
            f"scenario_batch_size={metadata.scenario_batch_size} "
            f"shuffle_train_scenarios={metadata.shuffle_train_scenarios}"
        ),
    )


def _run_loss_step(
    model: PortfolioAttentionModel,
    batch: dict[str, Any],
    loss_name: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    outputs = model(
        batch["x_stock"],
        batch["x_market"],
        batch["stock_indices"],
        target_returns=batch["r_stock"],
    )
    portfolio_returns = outputs["portfolio_return"]
    if portfolio_returns is None:
        raise RuntimeError("Training batch must provide target returns.")
    if portfolio_returns.ndim != 2:
        raise ValueError(
            "portfolio_returns must have shape [num_scenarios_in_batch, time_steps]. "
            f"Received {tuple(portfolio_returns.shape)}."
        )

    loss = build_loss(loss_name, portfolio_returns)
    scenario_final_returns = torch.prod(1.0 + portfolio_returns, dim=1) - 1.0
    summary = {
        "scenario_final_returns": scenario_final_returns,
        "scenario_mean_step_returns": portfolio_returns.mean(dim=1),
    }
    return loss, portfolio_returns, summary


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
    return max(1, int(select_best_from_last_x_epochs))


def _epoch_candidate_checkpoint_path(paths: PathsConfig, loss_name: str, epoch: int) -> Path:
    return paths.checkpoints_dir / f"train_candidate_{loss_name}_epoch_{epoch}.pt"


def _select_best_epoch_record(
    epoch_records: list[dict[str, Any]],
    select_best_from_last_x_epochs: int,
) -> dict[str, Any]:
    if not epoch_records:
        raise RuntimeError("No epoch records were collected for best-epoch selection.")
    candidate_records = epoch_records[-_normalize_best_epoch_selection_window(select_best_from_last_x_epochs) :]
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
    total_final_return = 0.0
    total_scenarios = 0

    for raw_batch in loader:
        batch = _move_batch_to_device(raw_batch, device)
        loss, _, summary = _run_loss_step(model, batch, loss_name)
        scenario_count = int(batch["x_stock"].shape[0])
        total_loss += float(loss.detach().cpu().item()) * scenario_count
        total_final_return += float(summary["scenario_final_returns"].mean().detach().cpu().item()) * scenario_count
        total_scenarios += scenario_count

    if total_scenarios == 0:
        raise RuntimeError("Evaluation loader produced no scenarios.")

    return total_loss / total_scenarios, total_final_return / total_scenarios


def run_epoch_training(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    paths: PathsConfig,
) -> dict[str, Any]:
    set_seed(train_config.seed)
    ensure_output_dirs(paths)
    device = resolve_device(train_config.device)
    log_path = paths.logs_dir / f"train_{train_config.loss_name}.log"
    _log_reproducibility_status(log_path, train_config, device)
    current_phase = "building_dataset"
    _write_training_status(
        paths,
        train_config.loss_name,
        "PREPARING_DATA",
        device=str(device),
        epoch=0,
        num_epochs=train_config.num_epochs,
        progress_ratio=0.0,
        phase=current_phase,
        message="Building dataset and scenario splits.",
    )

    dataset = PortfolioPanelDataset(data_config)
    train_dataset, validation_dataset, test_dataset = dataset.build_train_validation_test_datasets()
    if len(train_dataset) == 0 or len(validation_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError("Scenario training requires non-empty train, validation, and holdout test splits.")

    model = PortfolioAttentionModel(
        model_config,
        num_stocks=dataset.num_stocks,
        max_lookback=dataset.max_time_steps,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    generator = torch.Generator()
    generator.manual_seed(train_config.seed)
    scenario_batch_size = int(data_config.scenario_batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=scenario_batch_size,
        shuffle=bool(data_config.shuffle_train_scenarios),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=min(len(validation_dataset), scenario_batch_size),
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
            "Running scenario-based training with "
            f"train_scenarios={len(train_dataset)} validation_scenarios={len(validation_dataset)} "
            f"holdout_test_scenarios={len(test_dataset)} scenario_batch_size={scenario_batch_size} "
            f"num_epochs={train_config.num_epochs} "
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
    shape_logged = False

    current_phase = "training"
    _write_training_status(
        paths,
        train_config.loss_name,
        "RUNNING",
        device=str(device),
        epoch=0,
        num_epochs=train_config.num_epochs,
        progress_ratio=0.0,
        phase=current_phase,
        message="Dataset ready; waiting for first optimizer step.",
    )

    try:
        for epoch in range(1, train_config.num_epochs + 1):
            model.train()
            total_train_loss = 0.0
            total_train_final_return = 0.0
            total_train_scenarios = 0

            for raw_batch in train_loader:
                batch = _move_batch_to_device(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss, portfolio_returns, summary = _run_loss_step(model, batch, train_config.loss_name)
                if not shape_logged:
                    append_log(
                        log_path,
                        (
                            "Training tensor shapes: "
                            f"x_stock={tuple(batch['x_stock'].shape)} "
                            f"x_market={tuple(batch['x_market'].shape)} "
                            f"r_stock={tuple(batch['r_stock'].shape)} "
                            f"portfolio_returns={tuple(portfolio_returns.shape)}"
                        ),
                    )
                    shape_logged = True
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
                optimizer.step()

                scenario_count = int(batch["x_stock"].shape[0])
                total_train_loss += float(loss.detach().cpu().item()) * scenario_count
                total_train_final_return += float(
                    summary["scenario_final_returns"].mean().detach().cpu().item()
                ) * scenario_count
                total_train_scenarios += scenario_count

            if total_train_scenarios == 0:
                raise RuntimeError("Train loader produced no scenarios.")

            train_loss = total_train_loss / total_train_scenarios
            train_mean_final_return = total_train_final_return / total_train_scenarios
            val_loss, val_mean_final_return = _evaluate_epoch(
                model,
                validation_loader,
                device,
                train_config.loss_name,
            )

            epoch_metrics: dict[str, float | int | bool] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mean_final_return": train_mean_final_return,
                "val_loss": val_loss,
                "val_mean_final_return": val_mean_final_return,
            }
            history.append(dict(epoch_metrics))
            epochs_completed = epoch
            global_best_checkpoint_updated = False

            append_log(
                log_path,
                (
                    f"epoch={epoch} train_loss={train_loss:.8f} "
                    f"train_mean_final_return={train_mean_final_return:.8f} "
                    f"val_loss={val_loss:.8f} val_mean_final_return={val_mean_final_return:.8f}"
                ),
            )
            append_log(log_path, f"Aggregated validation loss at epoch {epoch}: {val_loss:.8f}")

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
                try:
                    Path(str(stale_record["checkpoint_path"])).unlink()
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
                train_mean_final_return=train_mean_final_return,
                val_loss=val_loss,
                val_mean_final_return=val_mean_final_return,
                best_epoch=current_window_best_epoch,
                best_val_loss=current_window_best_val_loss,
                global_best_val_loss=global_best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
                select_best_from_last_x_epochs=selection_window,
                phase=current_phase,
                message="Running optimizer and validation steps.",
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
    except Exception as exc:
        _write_training_status(
            paths,
            train_config.loss_name,
            "FAILED",
            error_message=str(exc),
            device=str(device),
            epoch=epochs_completed,
            num_epochs=train_config.num_epochs,
            progress_ratio=(epochs_completed / train_config.num_epochs) if train_config.num_epochs else 0.0,
            phase=current_phase,
            message="Training worker failed.",
        )
        raise

    if not epoch_selection_records:
        raise RuntimeError("Train loop did not record any epoch candidates for best selection.")

    selected_best_record = _select_best_epoch_record(epoch_selection_records, selection_window)
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

    append_log(log_path, f"Loading best checkpoint for final held-out evaluation: {best_checkpoint_path}.")
    current_phase = "evaluating"
    _write_training_status(
        paths,
        train_config.loss_name,
        "RUNNING",
        device=str(device),
        epoch=epochs_completed,
        num_epochs=train_config.num_epochs,
        progress_ratio=1.0,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        phase=current_phase,
        message="Running final held-out evaluation on the best checkpoint.",
    )
    final_backtest = run_evaluation(
        data_config=data_config,
        paths=paths,
        checkpoint_path=best_checkpoint_path,
        device_name=train_config.device,
    )
    append_log(
        log_path,
        (
            "Final held-out aggregate metrics: "
            f"mean_final_return={final_backtest['mean_final_return']:.8f} "
            f"std_final_return={final_backtest['std_final_return']:.8f} "
            f"median_final_return={final_backtest['median_final_return']:.8f} "
            f"worst_scenario_final_return={final_backtest['worst_scenario_final_return']:.8f}"
        ),
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
        "scenario_batch_size": scenario_batch_size,
        "num_epochs_requested": train_config.num_epochs,
        "epochs_completed": epochs_completed,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "select_best_from_last_x_epochs": train_config.select_best_from_last_x_epochs,
        "normalized_best_epoch_selection_window": selection_window,
        "effective_best_epoch_selection_window": effective_selection_window,
        "train_scenario_count": len(train_dataset),
        "validation_scenario_count": len(validation_dataset),
        "holdout_test_scenario_count": len(test_dataset),
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
        phase="completed",
        message="Training and evaluation finished successfully.",
    )

    return metrics


def run_training(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    paths: PathsConfig,
) -> dict[str, Any]:
    return run_epoch_training(data_config, model_config, train_config, paths)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run training for portfolio_attention.")
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument(
        "--loss",
        default=argparse.SUPPRESS,
        choices=["return", "terminal_return", "sharpe", "dsr", "sortino", "mdd", "cvar"],
    )
    parser.add_argument("--losses", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--grad-clip-norm", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--early-stopping-patience", type=int, default=argparse.SUPPRESS)
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

    train_overrides: dict[str, Any] = {}
    if "device" in args_dict:
        train_overrides["device"] = args_dict["device"]
    if "seed" in args_dict:
        train_overrides["seed"] = args_dict["seed"]
    if "loss" in args_dict:
        train_overrides["loss_name"] = args_dict["loss"]
    if "num_epochs" in args_dict:
        train_overrides["num_epochs"] = args_dict["num_epochs"]
    if "weight_decay" in args_dict:
        train_overrides["weight_decay"] = args_dict["weight_decay"]
    if "grad_clip_norm" in args_dict:
        train_overrides["grad_clip_norm"] = args_dict["grad_clip_norm"]
    if "early_stopping_patience" in args_dict:
        train_overrides["early_stopping_patience"] = args_dict["early_stopping_patience"]
    if "select_best_from_last_x_epochs" in args_dict:
        train_overrides["select_best_from_last_x_epochs"] = args_dict[
            "select_best_from_last_x_epochs"
        ]
    if train_overrides:
        resolved_train_config = replace(resolved_train_config, **train_overrides)

    return resolved_data_config, resolved_train_config


DEFAULT_LOSSES = ["return", "sharpe", "dsr", "sortino", "mdd", "cvar"]


def _normalize_losses(raw_losses: list[str]) -> list[str]:
    valid_losses = {"return", "sharpe", "dsr", "sortino", "mdd", "cvar"}
    result: list[str] = []
    seen: set[str] = set()
    for loss in raw_losses:
        normalized = loss.strip()
        if not normalized:
            continue
        if normalized == "terminal_return":
            normalized = "return"
        if normalized not in valid_losses:
            raise ValueError(f"Invalid loss: '{normalized}'. Must be one of {valid_losses} or 'terminal_return'")
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _parse_losses_args(args: argparse.Namespace) -> list[str]:
    args_dict = vars(args)
    if "loss" in args_dict:
        return _normalize_losses([args_dict["loss"]])
    if "losses" in args_dict:
        raw = args_dict["losses"]
        if not raw or not raw.strip():
            raise ValueError("--losses cannot be empty string")
        return _normalize_losses(raw.split(","))
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
        if arg in {"--losses", "--parallel"}:
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


def _preflight_runtime_config(data_config: DataConfig, train_config: TrainConfig, losses: list[str]) -> None:
    if not Path(data_config.scenario_dir).exists():
        raise FileNotFoundError(f"Scenario directory not found: {data_config.scenario_dir}")
    if train_config.num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, received {train_config.num_epochs}.")
    if not losses:
        raise ValueError("At least one loss must be requested.")


def _cleanup_previous_multi_loss_artifacts(paths: PathsConfig, losses: list[str]) -> None:
    if paths.status_dir.exists():
        for loss in losses:
            try:
                _status_path_for_loss(paths, loss).unlink()
            except FileNotFoundError:
                pass
    if paths.logs_dir.exists():
        for loss in losses:
            try:
                _console_log_path_for_loss(paths, loss).unlink()
            except FileNotFoundError:
                pass


def main() -> None:
    args = build_arg_parser().parse_args()
    args_dict = vars(args)
    paths = PathsConfig()

    parallel = args_dict.get("parallel", 1)
    if parallel < 1:
        raise ValueError("--parallel must be >= 1")

    losses_to_run = _parse_losses_args(args)
    worker_mode = _is_worker_mode()
    base_data_config, base_train_config = resolve_runtime_configs_from_args(args)

    if "loss" in args_dict:
        loss = losses_to_run[0]
        args.loss = loss
        data_config, train_config = resolve_runtime_configs_from_args(args)
        _preflight_runtime_config(data_config, train_config, [loss])

        if not worker_mode:
            print(f"\n>>> Running training with loss: {loss}")

        try:
            metrics = run_training(data_config, ModelConfig(), train_config, paths)
            if not worker_mode:
                print(f"--- Results for loss: {loss} ---")
                print(_format_terminal_summary(metrics))
        except Exception:
            if worker_mode:
                raise
            print(f"ERROR: Training for loss '{loss}' failed.")
            sys.exit(1)
        return

    _preflight_runtime_config(base_data_config, base_train_config, losses_to_run)
    gpu_ids = _resolve_round_robin_gpu_ids(parallel)
    ensure_output_dirs(paths)
    _cleanup_previous_multi_loss_artifacts(paths, losses_to_run)

    active_processes: list[dict[str, Any]] = []
    pending_losses = list(losses_to_run)
    failed_losses: list[str] = []
    failure_summaries: list[str] = []
    launch_index = 0

    env_base = os.environ.copy()
    env_base["PORTFOLIO_ATTENTION_CHILD"] = "1"

    status_rows = _status_snapshot(paths, losses_to_run)
    last_dashboard_signature = _dashboard_signature(status_rows)
    if _should_use_live_dashboard():
        console = Console(file=sys.stdout)
        live = Live(
            _render_multi_loss_dashboard(status_rows),
            console=console,
            auto_refresh=False,
            transient=False,
        )
        live.start()
        live.update(_render_multi_loss_dashboard(status_rows), refresh=True)
    else:
        live = None
        print(_render_multi_loss_dashboard(status_rows), flush=True)

    try:
        while pending_losses or active_processes:
            while pending_losses and len(active_processes) < parallel:
                loss = pending_losses.pop(0)
                gpu_id: int | None = None
                device_arg: str | None = None
                if gpu_ids:
                    gpu_id = gpu_ids[launch_index % len(gpu_ids)]
                    device_arg = f"cuda:{gpu_id}"
                cmd = _build_subprocess_cmd(loss, device=device_arg)
                env = env_base.copy()
                console_log_path = _console_log_path_for_loss(paths, loss)
                console_log_path.parent.mkdir(parents=True, exist_ok=True)
                console_handle = console_log_path.open("a", encoding="utf-8")
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=console_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                _write_training_status(
                    paths,
                    loss,
                    "STARTING",
                    pid=process.pid,
                    device=device_arg or "cpu",
                    epoch=0,
                    num_epochs=base_train_config.num_epochs,
                    progress_ratio=0.0,
                    phase="spawned",
                    message="Worker started; waiting for dataset setup.",
                )
                active_processes.append(
                    {
                        "loss": loss,
                        "gpu_id": gpu_id,
                        "process": process,
                        "console_handle": console_handle,
                        "console_log_path": console_log_path,
                    }
                )
                launch_index += 1
                time.sleep(0.2)

            remaining_processes: list[dict[str, Any]] = []
            for item in active_processes:
                process = item["process"]
                returncode = process.poll()
                if returncode is None:
                    remaining_processes.append(item)
                    continue

                item["console_handle"].close()
                if returncode not in (0, None):
                    failed_losses.append(str(item["loss"]))
                    status_data = _load_training_status(paths, str(item["loss"]))
                    if status_data.get("status") != "FAILED":
                        _write_training_status(
                            paths,
                            str(item["loss"]),
                            "FAILED",
                            pid=process.pid,
                            device=status_data.get("device", item.get("device", "-")),
                            epoch=int(status_data.get("epoch", 0)),
                            num_epochs=int(status_data.get("num_epochs", base_train_config.num_epochs)),
                            progress_ratio=float(status_data.get("progress_ratio", 0.0)),
                            phase=str(status_data.get("phase", "worker_exit")),
                            message="Worker process exited with a non-zero code.",
                            error_message=f"Worker exited with code {returncode}.",
                        )
                    failure_summaries.append(
                        _build_failure_summary(paths, str(item["loss"]), int(returncode))
                    )
            active_processes = remaining_processes

            status_rows = _status_snapshot(paths, losses_to_run)
            dashboard_signature = _dashboard_signature(status_rows)
            if live is not None:
                if dashboard_signature != last_dashboard_signature:
                    live.update(_render_multi_loss_dashboard(status_rows), refresh=True)
                    last_dashboard_signature = dashboard_signature
            elif dashboard_signature != last_dashboard_signature:
                print(_render_multi_loss_dashboard(status_rows), flush=True)
                last_dashboard_signature = dashboard_signature
            time.sleep(0.5)
    finally:
        for item in active_processes:
            try:
                item["console_handle"].close()
            except Exception:
                pass
        if live is not None:
            live.stop()

    if failed_losses:
        for summary in failure_summaries:
            print(summary, flush=True)
        print(f"ERROR: Some losses failed: {sorted(set(failed_losses))}")
        sys.exit(1)


if __name__ == "__main__":
    main()

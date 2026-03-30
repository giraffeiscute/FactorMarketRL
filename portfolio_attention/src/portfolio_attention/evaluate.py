"""Evaluation entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "portfolio_attention_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio_attention.config import (
        DataConfig,
        EvaluationConfig,
        ModelConfig,
        PathsConfig,
        TrainConfig,
    )
    from portfolio_attention.dataset import PortfolioPanelDataset
    from portfolio_attention.losses import sharpe_loss
    from portfolio_attention.model import PortfolioAttentionModel
    from portfolio_attention.utils import ensure_output_dirs, resolve_device, save_json
else:
    from .config import DataConfig, EvaluationConfig, ModelConfig, PathsConfig, TrainConfig
    from .dataset import PortfolioPanelDataset
    from .losses import sharpe_loss
    from .model import PortfolioAttentionModel
    from .utils import ensure_output_dirs, resolve_device, save_json

REQUIRED_AUX_COLUMNS = ["stock_id", "t", "mu", "alpha", "epsilon_variance"]
EXPORTED_TRAIN_CONFIG_KEYS = [
    "num_epochs",
    "weight_decay",
    "grad_clip_norm",
    "early_stopping_patience",
]
TERMINAL_OUTPUT_KEYS = [
    "state",
    "loss_name",
    "num_holdout_scenarios",
    "mean_final_return",
    "std_final_return",
    "median_final_return",
    "worst_scenario_final_return",
    "best_scenario_final_return",
    "best_scenario_id",
]


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _parse_source_time_to_index(raw_value: object) -> int:
    if isinstance(raw_value, str):
        match = re.fullmatch(r"t_(\d+)", raw_value.strip())
        if not match:
            raise ValueError(f"Unsupported source time label: {raw_value}")
        return int(match.group(1))
    return int(raw_value)


def _load_aux_frame(source_path: Path) -> pd.DataFrame:
    header = pq.read_schema(source_path).names
    missing_columns = [column for column in REQUIRED_AUX_COLUMNS if column not in header]
    if missing_columns:
        raise ValueError(
            "Evaluation export requires source panel columns: "
            f"{REQUIRED_AUX_COLUMNS}. Missing: {missing_columns}"
        )

    aux_frame = pd.read_parquet(source_path, columns=REQUIRED_AUX_COLUMNS)
    aux_frame["analysis_time_index"] = aux_frame["t"].map(_parse_source_time_to_index)
    return aux_frame


def _cleanup_stale_prediction_artifacts(output_dir: Path, loss_name: str) -> None:
    patterns = [
        f"*_{loss_name}_holdout_predictions.json",
        f"*_{loss_name}_allocation_pie.png",
        f"*_{loss_name}_allocation_bar.png",
        f"*_{loss_name}_all_stock_weights.csv",
        f"*_{loss_name}_best_weight_trajectory.png",
        f"{loss_name}_best_backtest_scenario_*.json",
        f"{loss_name}_best_backtest_scenario_*.png",
        f"{loss_name}_best_backtest_scenario_*.csv",
    ]
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _extract_exported_train_config(checkpoint: dict[str, Any]) -> dict[str, object]:
    checkpoint_train_config = checkpoint.get("train_config", {})
    checkpoint_data_config = checkpoint.get("data_config", {})
    exported = {
        key: checkpoint_train_config[key]
        for key in EXPORTED_TRAIN_CONFIG_KEYS
        if key in checkpoint_train_config
    }
    if "scenario_batch_size" in checkpoint_data_config:
        exported["scenario_batch_size"] = checkpoint_data_config["scenario_batch_size"]
    return exported


def _validate_checkpoint_metadata(checkpoint: dict[str, Any], dataset: PortfolioPanelDataset) -> None:
    checkpoint_metadata = checkpoint.get("metadata", {})
    checkpoint_num_stocks = checkpoint_metadata.get("selected_num_stocks")
    if checkpoint_num_stocks is not None and int(checkpoint_num_stocks) != dataset.num_stocks:
        raise ValueError(
            f"Checkpoint expects selected_num_stocks={checkpoint_num_stocks}, "
            f"but the evaluation dataset provides {dataset.num_stocks} stocks."
        )


def _format_terminal_summary(payload: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {payload[key]}" for key in TERMINAL_OUTPUT_KEYS if key in payload)


def _get_aux_lookup(aux_frame: pd.DataFrame) -> dict[tuple[str, int], dict[str, object]]:
    cached_lookup = aux_frame.attrs.get("_position_lookup")
    if cached_lookup is not None:
        return cached_lookup

    duplicated = aux_frame.duplicated(["stock_id", "analysis_time_index"], keep=False)
    if duplicated.any():
        duplicate_rows = (
            aux_frame.loc[duplicated, ["stock_id", "analysis_time_index"]]
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "Evaluation export found multiple source rows for the same "
            "(stock_id, analysis_time_index) keys. "
            f"Examples: {duplicate_rows}"
        )

    lookup: dict[tuple[str, int], dict[str, object]] = {}
    for row in aux_frame.itertuples(index=False):
        lookup[(str(row.stock_id), int(row.analysis_time_index))] = {
            "mu": row.mu,
            "alpha": row.alpha,
            "epsilon_variance": row.epsilon_variance,
        }
    aux_frame.attrs["_position_lookup"] = lookup
    return lookup


def enrich_positions(
    *,
    aux_frame: pd.DataFrame,
    analysis_time_index: int,
    positions: list[dict[str, object]],
) -> list[dict[str, object]]:
    aux_lookup = _get_aux_lookup(aux_frame)
    enriched: list[dict[str, object]] = []
    for rank, position in enumerate(positions, start=1):
        stock_id = str(position["stock_id"])
        match = aux_lookup.get((stock_id, analysis_time_index))
        if match is None:
            raise ValueError(
                f"Evaluation export could not find exactly one source row for stock_id={stock_id} "
                f"at analysis_time_index={analysis_time_index}."
            )
        enriched.append(
            {
                "rank": rank,
                "stock_id": stock_id,
                "weight": float(position["weight"]),
                "mu": match["mu"],
                "alpha": match["alpha"],
                "epsilon_variance": match["epsilon_variance"],
            }
        )
    return enriched


def enrich_top_k_positions(
    *,
    source_path: Path,
    metadata: dict[str, Any],
    top_positions: list[dict[str, object]],
) -> list[dict[str, object]]:
    analysis_time_index = int(metadata["analysis_time_index"])
    return enrich_positions(
        aux_frame=_load_aux_frame(source_path),
        analysis_time_index=analysis_time_index,
        positions=top_positions,
    )


def build_all_stock_positions(
    *,
    stock_ids: list[str],
    stock_weights: torch.Tensor,
) -> list[dict[str, object]]:
    positions = [
        {
            "stock_id": stock_id,
            "weight": float(weight),
        }
        for stock_id, weight in zip(stock_ids, stock_weights.tolist())
        if float(weight) > 0.0
    ]
    return sorted(positions, key=lambda item: item["weight"], reverse=True)


def group_allocations_by_state(all_stock_positions: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for position in all_stock_positions:
        key = (
            str(position["mu"]),
            str(position["epsilon_variance"]),
            str(position["alpha"]),
        )
        if key not in grouped:
            grouped[key] = {
                "mu": key[0],
                "epsilon_variance": key[1],
                "alpha": key[2],
                "total_weight": 0.0,
                "stock_count": 0,
            }
        grouped[key]["total_weight"] = float(grouped[key]["total_weight"]) + float(position["weight"])
        grouped[key]["stock_count"] = int(grouped[key]["stock_count"]) + 1
    return sorted(grouped.values(), key=lambda item: float(item["total_weight"]), reverse=True)


def append_cash_allocation(
    grouped_allocations: list[dict[str, object]],
    cash_weight: float,
) -> list[dict[str, object]]:
    if cash_weight < 0.0:
        raise ValueError(f"cash_weight must be non-negative, received {cash_weight}.")
    if cash_weight == 0.0:
        return list(grouped_allocations)
    return [
        *grouped_allocations,
        {
            "mu": "Cash",
            "epsilon_variance": "Cash",
            "alpha": "Cash",
            "total_weight": float(cash_weight),
            "stock_count": 0,
        },
    ]


def summarize_grouped_allocations(
    grouped_allocations: list[dict[str, object]],
    top_n: int = 10,
) -> list[dict[str, object]]:
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    cash_allocations = [item for item in grouped_allocations if str(item["mu"]) == "Cash"]
    non_cash_allocations = [item for item in grouped_allocations if str(item["mu"]) != "Cash"]
    if len(non_cash_allocations) <= top_n:
        return non_cash_allocations + cash_allocations

    head = non_cash_allocations[:top_n]
    tail = non_cash_allocations[top_n:]
    others = {
        "mu": "Others",
        "epsilon_variance": "Others",
        "alpha": "Others",
        "total_weight": float(sum(float(item["total_weight"]) for item in tail)),
        "stock_count": int(sum(int(item["stock_count"]) for item in tail)),
    }
    return head + [others] + cash_allocations


def _allocation_group_key(mu: object, epsilon_variance: object, alpha: object) -> tuple[str, str, str]:
    return (str(mu), str(epsilon_variance), str(alpha))


def format_allocation_group_label(grouped_allocation: dict[str, object]) -> str:
    mu = str(grouped_allocation["mu"])
    epsilon_variance = str(grouped_allocation["epsilon_variance"])
    alpha = str(grouped_allocation["alpha"])
    if mu in {"Cash", "Others"} and mu == epsilon_variance == alpha:
        return mu
    return f"mu={mu} | eps={epsilon_variance} | alpha={alpha}"


def _build_grouped_weight_trajectories(
    *,
    aux_frame: pd.DataFrame,
    analysis_time_index: int,
    stock_ids: list[str],
    stock_weights: torch.Tensor,
    cash_weights: torch.Tensor,
    grouped_allocations_top_n: list[dict[str, object]],
) -> list[dict[str, object]]:
    if stock_weights.ndim != 2:
        raise ValueError("stock_weights must have shape [T, N].")
    if cash_weights.ndim != 1:
        raise ValueError("cash_weights must have shape [T].")
    if stock_weights.shape[0] != cash_weights.shape[0]:
        raise ValueError(
            "stock_weights and cash_weights must share the same time dimension. "
            f"Received stock_weights.shape={tuple(stock_weights.shape)} and "
            f"cash_weights.shape={tuple(cash_weights.shape)}."
        )
    if stock_weights.shape[1] != len(stock_ids):
        raise ValueError(
            "stock_weights.shape[1] must match len(stock_ids). "
            f"Received {stock_weights.shape[1]} stocks and {len(stock_ids)} ids."
        )

    aux_lookup = _get_aux_lookup(aux_frame)
    zero_series = torch.zeros(stock_weights.shape[0], dtype=stock_weights.dtype, device=stock_weights.device)
    grouped_series: dict[tuple[str, str, str], torch.Tensor] = {}
    for index, stock_id in enumerate(stock_ids):
        match = aux_lookup.get((str(stock_id), analysis_time_index))
        if match is None:
            raise ValueError(
                f"Evaluation export could not find exactly one source row for stock_id={stock_id} "
                f"at analysis_time_index={analysis_time_index}."
            )
        key = _allocation_group_key(
            match["mu"],
            match["epsilon_variance"],
            match["alpha"],
        )
        if key not in grouped_series:
            grouped_series[key] = zero_series.clone()
        grouped_series[key] = grouped_series[key] + stock_weights[:, index]

    trajectories: list[dict[str, object]] = []
    for item in grouped_allocations_top_n:
        mu = str(item["mu"])
        if mu == "Cash":
            weights = cash_weights
        elif mu == "Others":
            continue
        else:
            key = _allocation_group_key(item["mu"], item["epsilon_variance"], item["alpha"])
            weights = grouped_series.get(key)
            if weights is None:
                raise ValueError(
                    "Grouped allocation summary referenced a group that is missing from the "
                    f"trajectory aggregation: {format_allocation_group_label(item)}."
                )
        trajectories.append(
            {
                "label": format_allocation_group_label(item),
                "weights": weights,
            }
        )
    return trajectories


def render_allocation_pie_chart(
    grouped_allocations: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    if not grouped_allocations:
        raise ValueError("Cannot render allocation pie chart without grouped allocations.")

    labels = [format_allocation_group_label(item) for item in grouped_allocations]
    sizes = [float(item["total_weight"]) for item in grouped_allocations]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(
        sizes,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2.0 else "",
        startangle=90,
        counterclock=False,
    )
    ax.set_title(title)
    ax.axis("equal")
    ax.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_allocation_bar_chart(
    grouped_allocations: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    if not grouped_allocations:
        raise ValueError("Cannot render allocation bar chart without grouped allocations.")

    labels = [format_allocation_group_label(item) for item in grouped_allocations]
    values = [float(item["total_weight"]) for item in grouped_allocations]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Total Weight")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_all_stock_weights_csv(
    all_stock_positions: list[dict[str, object]],
    output_path: Path,
) -> None:
    frame = pd.DataFrame(all_stock_positions)
    frame = frame.reindex(columns=["rank", "stock_id", "weight", "mu", "alpha", "epsilon_variance"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def export_allocation_artifacts(
    *,
    aux_frame: pd.DataFrame,
    analysis_time_index: int,
    stock_ids: list[str],
    stock_weights: torch.Tensor,
    cash_weight: float,
    portfolio_return: float,
    output_dir: Path,
    scenario_id: str,
    artifact_stem: str,
    allocation_group_top_n: int,
    loss_name: str,
) -> dict[str, object]:
    all_stock_positions = enrich_positions(
        aux_frame=aux_frame,
        analysis_time_index=analysis_time_index,
        positions=build_all_stock_positions(stock_ids=stock_ids, stock_weights=stock_weights),
    )
    grouped_allocations = append_cash_allocation(
        group_allocations_by_state(all_stock_positions),
        cash_weight,
    )
    grouped_allocations_top_n = summarize_grouped_allocations(
        grouped_allocations,
        top_n=allocation_group_top_n,
    )

    chart_title = (
        f"Top {allocation_group_top_n} Allocation Groups + Others + Cash: {scenario_id}\n"
        f"loss_name={loss_name} | portfolio_return={portfolio_return:.6f}"
    )
    pie_chart_path = output_dir / f"{artifact_stem}_allocation_pie.png"
    bar_chart_path = output_dir / f"{artifact_stem}_allocation_bar.png"
    all_stock_weights_csv_path = output_dir / f"{artifact_stem}_all_stock_weights.csv"

    save_all_stock_weights_csv(all_stock_positions, all_stock_weights_csv_path)
    render_allocation_pie_chart(grouped_allocations_top_n, pie_chart_path, chart_title)
    render_allocation_bar_chart(grouped_allocations_top_n, bar_chart_path, chart_title)

    return {
        "all_stock_weights": all_stock_positions,
        "all_stock_weights_csv": str(all_stock_weights_csv_path),
        "grouped_allocations": grouped_allocations,
        "grouped_allocations_top_n": grouped_allocations_top_n,
        "allocation_groups_top_n_plus_others": grouped_allocations_top_n,
        "allocation_group_top_n": allocation_group_top_n,
        "allocation_pie_chart": str(pie_chart_path),
        "allocation_bar_chart": str(bar_chart_path),
    }


def render_weight_trajectory_chart(
    *,
    scenario_id: str,
    grouped_weight_trajectories: list[dict[str, object]],
    target_time_indices: torch.Tensor,
    output_path: Path,
) -> None:
    if target_time_indices.ndim != 1:
        raise ValueError("target_time_indices must have shape [T].")
    if not grouped_weight_trajectories:
        raise ValueError("grouped_weight_trajectories must be non-empty.")

    fig, ax = plt.subplots(figsize=(14, 7))
    x_axis = target_time_indices.detach().cpu().numpy()
    for item in grouped_weight_trajectories:
        weights = item["weights"]
        if not isinstance(weights, torch.Tensor):
            raise ValueError("Each trajectory entry must provide a tensor in 'weights'.")
        linestyle = "--" if str(item["label"]) == "Cash" else "-"
        ax.plot(
            x_axis,
            weights.detach().cpu().numpy(),
            label=str(item["label"]),
            linestyle=linestyle,
            linewidth=2 if str(item["label"]) == "Cash" else 1.5,
        )
    ax.set_xlabel("Target Time Index")
    ax.set_ylabel("Weight")
    ax.set_title(f"Best Holdout Scenario Group Weight Trajectory: {scenario_id}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_per_scenario_payload(
    *,
    scenario_id: str,
    source_path: Path,
    loss_name: str,
    checkpoint: dict[str, Any],
    target_time_indices: torch.Tensor,
    portfolio_returns: torch.Tensor,
    stock_weights: torch.Tensor,
    cash_weights: torch.Tensor,
    dataset: PortfolioPanelDataset,
    evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    path_returns_cpu = portfolio_returns.detach().cpu()
    stock_weights_cpu = stock_weights.detach().cpu()
    cash_weights_cpu = cash_weights.detach().cpu()
    target_time_indices_cpu = target_time_indices.detach().cpu()

    final_return = float(torch.prod(1.0 + path_returns_cpu).item() - 1.0)
    final_cash_weight = float(cash_weights_cpu[-1].item())
    mean_cash_weight = float(cash_weights_cpu.mean().item())
    mean_step_return = float(path_returns_cpu.mean().item())
    std_step_return = float(path_returns_cpu.std(unbiased=False).item())

    top_k = min(5, dataset.num_stocks)
    final_stock_weights = stock_weights_cpu[-1]
    top_values, top_indices = torch.topk(final_stock_weights, k=top_k)
    top_positions = [
        {
            "stock_id": dataset.selected_stock_ids[int(index)],
            "weight": float(weight.item()),
        }
        for weight, index in zip(top_values, top_indices)
    ]

    analysis_time_index = int(target_time_indices_cpu[-1].item())
    aux_frame = _load_aux_frame(source_path)
    enriched_top_positions = enrich_positions(
        aux_frame=aux_frame,
        analysis_time_index=analysis_time_index,
        positions=top_positions,
    )

    payload: dict[str, Any] = {
        "scenario_id": scenario_id,
        "source_path": str(source_path),
        "loss_name": loss_name,
        "state": dataset.state,
        "evaluation_split": "holdout_test",
        "train_config": _extract_exported_train_config(checkpoint),
        "final_return": final_return,
        "mean_step_return": mean_step_return,
        "std_step_return": std_step_return,
        "final_cash_weight": final_cash_weight,
        "mean_cash_weight": mean_cash_weight,
        "num_time_steps": int(path_returns_cpu.shape[0]),
        "analysis_time_index": analysis_time_index,
        "feature_time_start_index": int(target_time_indices_cpu[0].item()) - 1,
        "feature_time_end_index": int(target_time_indices_cpu[-1].item()) - 1,
        "target_time_start_index": int(target_time_indices_cpu[0].item()),
        "target_time_end_index": int(target_time_indices_cpu[-1].item()),
        "top_k_stock_weights": enriched_top_positions,
        "allocation_group_top_n": evaluation_config.allocation_group_top_n,
    }
    if loss_name == "sharpe":
        payload["sharpe_like"] = float((-sharpe_loss(path_returns_cpu).item()))

    payload["_final_stock_weights_tensor"] = final_stock_weights
    payload["_stock_weights_tensor"] = stock_weights_cpu
    payload["_cash_weights_tensor"] = cash_weights_cpu
    payload["_target_time_indices_tensor"] = target_time_indices_cpu
    return payload


def _export_best_backtest_payload(
    *,
    best_payload: dict[str, Any],
    checkpoint: dict[str, Any],
    dataset: PortfolioPanelDataset,
    output_dir: Path,
    evaluation_config: EvaluationConfig,
    loss_name: str,
) -> dict[str, Any]:
    scenario_id = str(best_payload["scenario_id"])
    source_path = Path(str(best_payload["source_path"]))
    artifact_stem = f"{loss_name}_best_backtest_scenario"
    aux_frame = _load_aux_frame(source_path)
    allocation_payload = export_allocation_artifacts(
        aux_frame=aux_frame,
        analysis_time_index=int(best_payload["analysis_time_index"]),
        stock_ids=dataset.selected_stock_ids,
        stock_weights=best_payload["_final_stock_weights_tensor"],
        cash_weight=float(best_payload["final_cash_weight"]),
        portfolio_return=float(best_payload["final_return"]),
        output_dir=output_dir,
        scenario_id=scenario_id,
        artifact_stem=artifact_stem,
        allocation_group_top_n=evaluation_config.allocation_group_top_n,
        loss_name=loss_name,
    )

    prediction_json_path = output_dir / f"{artifact_stem}_prediction.json"
    exported_payload = {
        key: value
        for key, value in best_payload.items()
        if not key.startswith("_")
    }
    exported_payload.update(
        {
            "best_backtest_scenario": True,
            "train_config": _extract_exported_train_config(checkpoint),
            "all_stock_weights": allocation_payload["all_stock_weights"],
            "all_stock_weights_csv": allocation_payload["all_stock_weights_csv"],
            "allocation_groups": allocation_payload["grouped_allocations"],
            "grouped_allocations_top_n": allocation_payload["grouped_allocations_top_n"],
            "allocation_groups_top_n_plus_others": allocation_payload[
                "allocation_groups_top_n_plus_others"
            ],
            "allocation_group_top_n": allocation_payload["allocation_group_top_n"],
            "allocation_pie_chart": allocation_payload["allocation_pie_chart"],
            "allocation_bar_chart": allocation_payload["allocation_bar_chart"],
        }
    )
    save_json(exported_payload, prediction_json_path)

    return {
        "payload": exported_payload,
        "prediction_json_path": str(prediction_json_path),
        "all_stock_weights_csv": str(allocation_payload["all_stock_weights_csv"]),
        "grouped_allocations_top_n": allocation_payload["grouped_allocations_top_n"],
        "allocation_pie_chart": str(allocation_payload["allocation_pie_chart"]),
        "allocation_bar_chart": str(allocation_payload["allocation_bar_chart"]),
    }


def run_evaluation(
    data_config: DataConfig,
    paths: PathsConfig,
    checkpoint_path: Path | None = None,
    device_name: str = "auto",
    top_k: int = 5,
    evaluation_config: EvaluationConfig | None = None,
    loss_name: str | None = None,
) -> dict[str, Any]:
    del top_k
    ensure_output_dirs(paths)
    device = resolve_device(device_name)
    resolved_evaluation_config = evaluation_config or EvaluationConfig()
    dataset = PortfolioPanelDataset(data_config)
    holdout_dataset = dataset.get_split_dataset("test")
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=min(len(holdout_dataset), data_config.scenario_batch_size),
        shuffle=False,
    )

    resolved_checkpoint = checkpoint_path or (
        paths.checkpoints_dir / TrainConfig(loss_name=loss_name or "dsr").train_best_checkpoint_name
    )
    checkpoint = torch.load(resolved_checkpoint, map_location=device, weights_only=False)
    _validate_checkpoint_metadata(checkpoint, dataset)

    checkpoint_model_config = checkpoint["model_config"]
    max_lookback = checkpoint.get("max_lookback")
    if max_lookback is None:
        max_lookback = checkpoint.get("metadata", {}).get(
            "train_segment_time_steps",
            dataset.max_time_steps,
        )
    filtered_config_dict = {
        key: value
        for key, value in checkpoint_model_config.items()
        if key in ModelConfig.__dataclass_fields__
    }
    model_config = ModelConfig(**filtered_config_dict)
    model = PortfolioAttentionModel(
        model_config,
        num_stocks=dataset.num_stocks,
        max_lookback=int(max_lookback),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    checkpoint_train_config = checkpoint.get("train_config", {})
    checkpoint_loss_name = str(checkpoint_train_config.get("loss_name", loss_name or "unknown")).lower()
    state_predictions_dir = paths.get_state_predictions_dir(dataset.state)
    state_predictions_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_prediction_artifacts(state_predictions_dir, checkpoint_loss_name)
    legacy_holdout_dir = state_predictions_dir / "holdout_test"
    if legacy_holdout_dir.exists():
        _cleanup_stale_prediction_artifacts(legacy_holdout_dir, checkpoint_loss_name)

    per_scenario_payloads: list[dict[str, Any]] = []
    for raw_batch in holdout_loader:
        batch = _move_batch_to_device(raw_batch, device)
        with torch.no_grad():
            outputs = model(
                batch["x_stock"],
                batch["x_market"],
                batch["stock_indices"],
                target_returns=batch["r_stock"],
            )

        if outputs["portfolio_return"] is None:
            raise RuntimeError("Evaluation batch must provide target returns.")

        path_returns = outputs["portfolio_return"]
        stock_weights = outputs["stock_weights"]
        cash_weights = outputs["cash_weight"]
        scenario_ids = list(batch["scenario_id"])
        source_paths = [Path(value) for value in batch["source_path"]]
        target_time_indices = batch["target_time_indices"]

        for index, scenario_id in enumerate(scenario_ids):
            per_scenario_payloads.append(
                _build_per_scenario_payload(
                    scenario_id=scenario_id,
                    source_path=source_paths[index],
                    loss_name=checkpoint_loss_name,
                    checkpoint=checkpoint,
                    target_time_indices=target_time_indices[index],
                    portfolio_returns=path_returns[index],
                    stock_weights=stock_weights[index],
                    cash_weights=cash_weights[index],
                    dataset=dataset,
                    evaluation_config=resolved_evaluation_config,
                )
            )

    if len(per_scenario_payloads) != len(holdout_dataset):
        raise RuntimeError(
            "Holdout evaluation did not produce a per-scenario payload for every holdout scenario."
        )

    final_returns = np.asarray(
        [float(item["final_return"]) for item in per_scenario_payloads],
        dtype=np.float64,
    )
    best_index = int(final_returns.argmax())
    worst_index = int(final_returns.argmin())
    best_payload = per_scenario_payloads[best_index]
    worst_payload = per_scenario_payloads[worst_index]

    best_export = _export_best_backtest_payload(
        best_payload=best_payload,
        checkpoint=checkpoint,
        dataset=dataset,
        output_dir=state_predictions_dir,
        evaluation_config=resolved_evaluation_config,
        loss_name=checkpoint_loss_name,
    )
    best_weight_plot_path = (
        state_predictions_dir
        / f"{checkpoint_loss_name}_best_backtest_scenario_weight_trajectory.png"
    )
    best_aux_frame = _load_aux_frame(Path(str(best_payload["source_path"])))
    grouped_weight_trajectories = _build_grouped_weight_trajectories(
        aux_frame=best_aux_frame,
        analysis_time_index=int(best_payload["analysis_time_index"]),
        stock_ids=dataset.selected_stock_ids,
        stock_weights=best_payload["_stock_weights_tensor"],
        cash_weights=best_payload["_cash_weights_tensor"],
        grouped_allocations_top_n=list(best_export["grouped_allocations_top_n"]),
    )
    render_weight_trajectory_chart(
        scenario_id=str(best_payload["scenario_id"]),
        grouped_weight_trajectories=grouped_weight_trajectories,
        target_time_indices=best_payload["_target_time_indices_tensor"],
        output_path=best_weight_plot_path,
    )

    per_scenario_rows = [
        {
            "scenario_id": item["scenario_id"],
            "source_path": item["source_path"],
            "final_return": item["final_return"],
            "mean_step_return": item["mean_step_return"],
            "std_step_return": item["std_step_return"],
            "final_cash_weight": item["final_cash_weight"],
            "mean_cash_weight": item["mean_cash_weight"],
        }
        for item in per_scenario_payloads
    ]
    per_scenario_csv_path = paths.metrics_dir / f"evaluation_metrics_{checkpoint_loss_name}_per_scenario.csv"
    pd.DataFrame(per_scenario_rows).to_csv(per_scenario_csv_path, index=False)

    aggregate_payload: dict[str, Any] = {
        "state": dataset.state,
        "loss_name": checkpoint_loss_name,
        "evaluation_split": "holdout_test",
        "num_holdout_scenarios": len(per_scenario_payloads),
        "mean_final_return": float(final_returns.mean()),
        "std_final_return": float(final_returns.std(ddof=0)),
        "median_final_return": float(np.median(final_returns)),
        "worst_scenario_final_return": float(final_returns.min()),
        "best_scenario_final_return": float(final_returns.max()),
        "best_scenario_id": best_payload["scenario_id"],
        "worst_scenario_id": worst_payload["scenario_id"],
        "best_backtest_scenario_source_path": best_payload["source_path"],
        "per_scenario_metrics_csv": str(per_scenario_csv_path),
        "best_backtest_scenario_prediction_file": best_export["prediction_json_path"],
        "best_backtest_scenario_all_stock_weights_csv": best_export["all_stock_weights_csv"],
        "best_backtest_scenario_allocation_pie_chart": best_export["allocation_pie_chart"],
        "best_backtest_scenario_allocation_bar_chart": best_export["allocation_bar_chart"],
        "best_backtest_scenario_weight_trajectory_chart": str(best_weight_plot_path),
        "best_scenario_weight_trajectory_chart": str(best_weight_plot_path),
        "train_config": _extract_exported_train_config(checkpoint),
        "metadata": dataset.metadata.as_dict(),
    }
    save_json(aggregate_payload, paths.metrics_dir / f"evaluation_metrics_{checkpoint_loss_name}.json")

    for item in per_scenario_payloads:
        item.pop("_final_stock_weights_tensor", None)
        item.pop("_stock_weights_tensor", None)
        item.pop("_cash_weights_tensor", None)
        item.pop("_target_time_indices_tensor", None)

    return aggregate_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evaluation for portfolio_attention.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--loss",
        default=None,
        choices=["return", "sharpe", "dsr", "sortino", "mdd", "cvar"],
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = PathsConfig()
    payload = run_evaluation(
        data_config=DataConfig(),
        paths=paths,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
        loss_name=args.loss,
    )
    print(_format_terminal_summary(payload))


if __name__ == "__main__":
    main()

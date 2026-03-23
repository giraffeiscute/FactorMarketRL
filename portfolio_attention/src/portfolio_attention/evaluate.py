"""Diagnostic evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portfolio_attention.config import DataConfig, ModelConfig, PathsConfig, TrainConfig
    from portfolio_attention.dataset import PortfolioPanelDataset
    from portfolio_attention.losses import sharpe_loss
    from portfolio_attention.model import PortfolioAttentionModel
    from portfolio_attention.utils import (
        ensure_output_dirs,
        resolve_device,
        save_json,
        state_id_from_csv_path,
    )
else:
    from .config import DataConfig, ModelConfig, PathsConfig, TrainConfig
    from .dataset import PortfolioPanelDataset
    from .losses import sharpe_loss
    from .model import PortfolioAttentionModel
    from .utils import ensure_output_dirs, resolve_device, save_json, state_id_from_csv_path

REQUIRED_AUX_COLUMNS = ["stock_id", "t", "mu", "alpha", "epsilon_variance"]
EXPORTED_TRAIN_CONFIG_KEYS = [
    "batch_size",
    "num_epochs",
    "weight_decay",
    "grad_clip_norm",
    "early_stopping_patience",
]


def _parse_source_time_to_index(raw_value: object) -> int:
    if isinstance(raw_value, str):
        match = re.fullmatch(r"t_(\d+)", raw_value.strip())
        if not match:
            raise ValueError(f"Unsupported source time label: {raw_value}")
        return int(match.group(1))
    return int(raw_value)


def _load_aux_frame(source_csv_path: Path) -> pd.DataFrame:
    header = pd.read_csv(source_csv_path, nrows=0).columns.tolist()
    missing_columns = [column for column in REQUIRED_AUX_COLUMNS if column not in header]
    if missing_columns:
        raise ValueError(
            "Diagnostic export requires source CSV columns: "
            f"{REQUIRED_AUX_COLUMNS}. Missing: {missing_columns}"
        )

    aux_frame = pd.read_csv(source_csv_path, usecols=REQUIRED_AUX_COLUMNS)
    aux_frame["analysis_time_index"] = aux_frame["t"].map(_parse_source_time_to_index)
    return aux_frame


def _extract_exported_train_config(checkpoint: dict) -> dict[str, object]:
    checkpoint_train_config = checkpoint.get("train_config", {})
    return {
        key: checkpoint_train_config[key]
        for key in EXPORTED_TRAIN_CONFIG_KEYS
        if key in checkpoint_train_config
    }


def _validate_checkpoint_metadata(checkpoint: dict, dataset: PortfolioPanelDataset) -> None:
    checkpoint_metadata = checkpoint.get("metadata", {})
    checkpoint_num_stocks = checkpoint_metadata.get("selected_num_stocks")
    if checkpoint_num_stocks is not None and int(checkpoint_num_stocks) != dataset.num_stocks:
        raise ValueError(
            f"Checkpoint expects selected_num_stocks={checkpoint_num_stocks}, "
            f"but the evaluation dataset provides {dataset.num_stocks} stocks."
        )


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
            "Diagnostic export found multiple source rows for the same "
            "(stock_id, test_horizon_start_index) keys. "
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
    metadata: dict,
    positions: list[dict[str, object]],
) -> list[dict]:
    analysis_time_index = int(metadata["test_horizon_start_index"])
    aux_lookup = _get_aux_lookup(aux_frame)
    enriched: list[dict] = []
    for rank, position in enumerate(positions, start=1):
        stock_id = str(position["stock_id"])
        match = aux_lookup.get((stock_id, analysis_time_index))
        if match is None:
            raise ValueError(
                f"Diagnostic export could not find exactly one source row for stock_id={stock_id} "
                f"at test_horizon_start_index={analysis_time_index}."
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
    source_csv_path: Path,
    metadata: dict,
    top_positions: list[dict[str, object]],
) -> list[dict]:
    return enrich_positions(
        aux_frame=_load_aux_frame(source_csv_path),
        metadata=metadata,
        positions=top_positions,
    )


def build_all_stock_positions(
    *,
    stock_ids: list[str],
    stock_weights: torch.Tensor,
) -> list[dict]:
    positions = [
        {
            "stock_id": stock_id,
            "weight": float(weight),
        }
        for stock_id, weight in zip(stock_ids, stock_weights.tolist())
        if float(weight) > 0.0
    ]
    return sorted(positions, key=lambda item: item["weight"], reverse=True)


def group_allocations_by_state(all_stock_positions: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], dict] = {}
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
        grouped[key]["total_weight"] += float(position["weight"])
        grouped[key]["stock_count"] += 1

    return sorted(grouped.values(), key=lambda item: item["total_weight"], reverse=True)


def summarize_grouped_allocations(
    grouped_allocations: list[dict],
    top_n: int = 10,
) -> list[dict]:
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    if len(grouped_allocations) <= top_n:
        return grouped_allocations

    head = grouped_allocations[:top_n]
    tail = grouped_allocations[top_n:]
    others = {
        "mu": "Others",
        "epsilon_variance": "Others",
        "alpha": "Others",
        "total_weight": float(sum(float(item["total_weight"]) for item in tail)),
        "stock_count": int(sum(int(item["stock_count"]) for item in tail)),
    }
    return head + [others]


def render_allocation_pie_chart(
    grouped_allocations: list[dict],
    output_path: Path,
    title: str,
) -> None:
    if not grouped_allocations:
        raise ValueError("Cannot render allocation pie chart without grouped allocations.")

    labels = [
        f"mu={item['mu']} | eps={item['epsilon_variance']} | alpha={item['alpha']}"
        for item in grouped_allocations
    ]
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
    grouped_allocations: list[dict],
    output_path: Path,
    title: str,
) -> None:
    if not grouped_allocations:
        raise ValueError("Cannot render allocation bar chart without grouped allocations.")

    labels = [
        f"mu={item['mu']}\neps={item['epsilon_variance']}\nalpha={item['alpha']}"
        for item in grouped_allocations
    ]
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
    all_stock_positions: list[dict],
    output_path: Path,
) -> None:
    frame = pd.DataFrame(all_stock_positions)
    frame = frame.reindex(columns=["rank", "stock_id", "weight", "mu", "alpha", "epsilon_variance"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


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

    resolved_checkpoint = checkpoint_path or (paths.checkpoints_dir / TrainConfig().checkpoint_name)
    checkpoint = torch.load(resolved_checkpoint, map_location=device, weights_only=False)
    _validate_checkpoint_metadata(checkpoint, dataset)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = PortfolioAttentionModel(model_config, num_stocks=dataset.num_stocks).to(device)
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
            "stock_id": dataset.selected_stock_ids[int(index)],
            "weight": float(weight.item()),
        }
        for weight, index in zip(top_values, top_indices)
    ]

    source_csv_path = Path(data_config.csv_path)
    aux_frame = _load_aux_frame(source_csv_path)
    enriched_top_positions = enrich_positions(
        aux_frame=aux_frame,
        metadata=dataset.metadata.as_dict(),
        positions=top_positions,
    )
    all_stock_positions = enrich_positions(
        aux_frame=aux_frame,
        metadata=dataset.metadata.as_dict(),
        positions=build_all_stock_positions(
            stock_ids=dataset.selected_stock_ids,
            stock_weights=stock_weights,
        ),
    )
    grouped_allocations = group_allocations_by_state(all_stock_positions)
    grouped_allocations_top10 = summarize_grouped_allocations(grouped_allocations, top_n=10)
    state_id = state_id_from_csv_path(source_csv_path)
    pie_chart_path = paths.outputs_dir / f"{state_id}_allocation_pie.png"
    bar_chart_path = paths.outputs_dir / f"{state_id}_allocation_bar.png"
    all_stock_weights_csv_path = paths.predictions_dir / f"{state_id}_all_stock_weights.csv"
    save_all_stock_weights_csv(all_stock_positions, all_stock_weights_csv_path)
    render_allocation_pie_chart(
        grouped_allocations=grouped_allocations_top10,
        output_path=pie_chart_path,
        title=f"Top 10 Allocation Groups + Others: {state_id}",
    )
    render_allocation_bar_chart(
        grouped_allocations=grouped_allocations_top10,
        output_path=bar_chart_path,
        title=f"Top 10 Allocation Groups + Others: {state_id}",
    )

    prediction_payload = {
        "source_path": state_id,
        "diagnostic_only": True,
        "device": str(device),
        "train_config": _extract_exported_train_config(checkpoint),
        "portfolio_return": portfolio_return,
        "average_portfolio_return": portfolio_return,
        "cash_weight": cash_weight,
        "average_cash_weight": cash_weight,
        "sharpe_like": sharpe_like,
        "metadata": {
            key: value
            for key, value in dataset.metadata.as_dict().items()
            if key != "source_path"
        },
        "top_k_stock_weights": enriched_top_positions,
    }
    metrics_payload = {
        **prediction_payload,
        "all_stock_weights": all_stock_positions,
        "all_stock_weights_csv": str(all_stock_weights_csv_path),
        "allocation_groups": grouped_allocations,
        "allocation_groups_top10_plus_others": grouped_allocations_top10,
        "allocation_pie_chart": str(pie_chart_path),
        "allocation_bar_chart": str(bar_chart_path),
    }
    legacy_csv_path = paths.predictions_dir / "diagnostic_predictions.csv"
    if legacy_csv_path.exists():
        legacy_csv_path.unlink()
    save_json(prediction_payload, paths.predictions_dir / "diagnostic_predictions.json")
    save_json(metrics_payload, paths.metrics_dir / "evaluation_metrics.json")
    return prediction_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run diagnostic evaluation for portfolio_attention.")
    parser.add_argument("--mode", default="diagnostic")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-stocks", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = PathsConfig()
    default_data_config = DataConfig()
    data_config = DataConfig(
        csv_path=args.data_path or default_data_config.csv_path,
        num_stocks=args.num_stocks,
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

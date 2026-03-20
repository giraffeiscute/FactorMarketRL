"""Configuration objects for portfolio_attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return project_root().parent


def default_data_path() -> Path:
    return repo_root() / "toy_ff_generator" / "outputs" / "bull_4860_81_panel_long.csv"


@dataclass
class PathsConfig:
    project_dir: Path = field(default_factory=project_root)
    repo_dir: Path = field(default_factory=repo_root)
    output_root: Path | None = None

    @property
    def outputs_dir(self) -> Path:
        return self.output_root or self.project_dir / "outputs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.outputs_dir / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.outputs_dir / "metrics"

    @property
    def logs_dir(self) -> Path:
        return self.outputs_dir / "logs"

    @property
    def predictions_dir(self) -> Path:
        return self.outputs_dir / "predictions"


@dataclass
class DataConfig:
    """Dataset construction settings, including the fixed stock count."""

    csv_path: Path = field(default_factory=default_data_path)
    lookback: int = 60
    train_ratio: float = 0.75
    analysis_entry_day: int | None = None
    analysis_exit_day: int | None = None
    num_stocks: int | None = None

    def resolved_entry_day(self) -> int:
        return self.analysis_entry_day or (self.lookback + 1)


@dataclass
class ModelConfig:
    num_stocks: int = 4860
    stock_feature_dim: int = 4
    market_feature_dim: int = 3
    lookback: int = 60
    stock_temporal_dim: int = 16
    market_temporal_dim: int = 16
    cross_sectional_dim: int = 16
    stock_id_embedding_dim: int = 8
    attention_heads: int = 1
    dropout: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    """Training-loop settings only. Stock count lives in DataConfig."""

    seed: int = 7
    learning_rate: float = 1e-3
    batch_size: int = 16
    num_epochs: int = 30
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    diagnostic_steps: int = 1
    loss_name: str = "return"
    mode: str = "diagnostic"
    device: str = "auto"
    checkpoint_name: str = "diagnostic_last.pt"

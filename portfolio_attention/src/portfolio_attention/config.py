"""Configuration objects for portfolio_attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import math


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return project_root().parent


def default_scenario_dir() -> Path:
    return repo_root() / "toy_ff_generator" / "outputs" / "data v3" / "bull"


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

    @property
    def status_dir(self) -> Path:
        return self.outputs_dir / "status"

    def get_state_predictions_dir(self, state: str) -> Path:
        return self.predictions_dir / state

    def get_scenario_predictions_dir(self, state_id: str) -> Path:
        """Backward-compatible helper for paths keyed by a scenario/state id."""
        return self.get_state_predictions_dir(state_id.split("_")[0])


@dataclass
class DataConfig:
    """Scenario-aware dataset construction settings.

    The project is scenario-only. One scenario file corresponds to one scenario path.
    The loader deterministically splits scenario files into train / validation /
    holdout-test groups, then splits each scenario internally across time.
    """

    scenario_dir: Path = field(default_factory=default_scenario_dir)
    scenario_glob: str = "{state}_*_PL_*.parquet"

    num_train_scenarios: int = 54
    num_validation_scenarios: int = 8
    num_test_scenarios: int = 2

    scenario_train_split_ratio: float = 0.70
    scenario_validation_split_ratio: float = 0.15
    scenario_test_split_ratio: float = 0.15

    # Number of full scenarios processed per optimizer step.
    scenario_batch_size: int = 8
    shuffle_train_scenarios: bool = True
    shuffle_train_scenarios_seed: int = 2

    # Fixed or maximum stock universe size to keep from each scenario.
    num_stocks: int = 4860

    def __post_init__(self) -> None:
        scenario_counts = {
            "num_train_scenarios": int(self.num_train_scenarios),
            "num_validation_scenarios": int(self.num_validation_scenarios),
            "num_test_scenarios": int(self.num_test_scenarios),
        }
        for name, value in scenario_counts.items():
            if value <= 0:
                raise ValueError(f"DataConfig.{name} must be positive, received {value}.")

        if int(self.scenario_batch_size) <= 0:
            raise ValueError(
                "DataConfig.scenario_batch_size must be positive, "
                f"received {self.scenario_batch_size}."
            )

        if self.shuffle_train_scenarios_seed is not None:
            resolved_shuffle_seed = int(self.shuffle_train_scenarios_seed)
            if resolved_shuffle_seed < 0:
                raise ValueError(
                    "DataConfig.shuffle_train_scenarios_seed must be non-negative, "
                    f"received {self.shuffle_train_scenarios_seed}."
                )
            self.shuffle_train_scenarios_seed = resolved_shuffle_seed

        ratio_fields = {
            "scenario_train_split_ratio": float(self.scenario_train_split_ratio),
            "scenario_validation_split_ratio": float(self.scenario_validation_split_ratio),
            "scenario_test_split_ratio": float(self.scenario_test_split_ratio),
        }
        for name, value in ratio_fields.items():
            if value <= 0.0:
                raise ValueError(f"DataConfig.{name} must be positive, received {value}.")

        ratio_sum = sum(ratio_fields.values())
        if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "Scenario time split ratios must sum to 1.0. "
                f"Received train={self.scenario_train_split_ratio}, "
                f"validation={self.scenario_validation_split_ratio}, "
                f"test={self.scenario_test_split_ratio}, sum={ratio_sum:.12f}."
            )

        if self.expected_total_scenarios <= 0:
            raise ValueError(
                "Expected total scenarios must be positive, "
                f"received {self.expected_total_scenarios}."
            )

    @property
    def expected_total_scenarios(self) -> int:
        return (
            int(self.num_train_scenarios)
            + int(self.num_validation_scenarios)
            + int(self.num_test_scenarios)
        )


@dataclass
class ModelConfig:
    """Scenario-mode model settings.

    The defaults are intentionally lightweight enough to preserve a full
    `[scenario, time, stock]` layout during training on large stock universes.
    """

    stock_feature_dim: int = 4
    market_feature_dim: int = 3
    stock_temporal_dim: int = 64
    market_temporal_dim: int = 32
    cross_sectional_dim: int = 64
    stock_id_embedding_dim: int = 32
    time_positional_encoding_type: str = "sinusoidal"
    attention_heads: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        valid_time_positional_encoding_types = {"running_mean", "sinusoidal"}
        if self.time_positional_encoding_type not in valid_time_positional_encoding_types:
            raise ValueError(
                "ModelConfig.time_positional_encoding_type must be one of "
                f"{sorted(valid_time_positional_encoding_types)}, "
                f"received {self.time_positional_encoding_type!r}."
            )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    """Training settings for scenario-mode optimization."""

    seed: int = 42
    learning_rate: float = 1e-4
    num_epochs: int = 300
    weight_decay: float = 1e-3
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 100
    select_best_from_last_x_epochs: int = 100
    loss_name: str = ""
    device: str = "auto"

    def _checkpoint_name(self, stem: str) -> str:
        if self.loss_name:
            return f"{stem}_{self.loss_name}.pt"
        return f"{stem}.pt"

    @property
    def train_best_checkpoint_name(self) -> str:
        return self._checkpoint_name("train_best")

    @property
    def train_last_checkpoint_name(self) -> str:
        return self._checkpoint_name("train_last")


@dataclass
class EvaluationConfig:
    """Evaluation and visualization settings."""

    allocation_group_top_n: int = 7

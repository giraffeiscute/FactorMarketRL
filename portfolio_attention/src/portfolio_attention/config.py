"""Configuration objects for portfolio_attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return project_root().parent


def default_data_path() -> Path:
    return repo_root() / "toy_ff_generator" / "outputs" / "data v1" / "bull_4860_200_panel_long.csv"


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

    def get_scenario_predictions_dir(self, state_id: str) -> Path:
        """依據 state_id (格式 {S}_{N}_{T}) 的前綴 S 取得對應的 predictions 子目錄。"""
        scenario = state_id.split("_")[0]
        return self.predictions_dir / scenario


@dataclass
class DataConfig:
    """資料集建構設定。

    `num_stocks` 是專案中明確指定股票數量的主要參數。
    若未特別覆寫，則使用資料本身的股票數量。
    lookback 不再是固定全域設定，而是依 train / validation / backtest
    三段 split 邊界與 `analysis_horizon_days` 動態決定。
    """

    # 輸入資料 CSV 檔案路徑。
    csv_path: Path = field(default_factory=default_data_path)

    # train split 所佔比例，例如 0.70 表示 70% 資料用於訓練參數更新。
    train_split_ratio: float = 0.80

    # validation split 所佔比例，用來選 best epoch 與 early stopping。
    validation_split_ratio: float = 0.1

    # backtest split 所佔比例，只在訓練完成後做一次最終回測。
    backtest_split_ratio: float = 0.1

    # 分析 horizon 天數，用來定義每筆樣本的未來報酬觀察區間。
    analysis_horizon_days: int = 19

    # 使用的股票總數上限或目標數量。
    num_stocks: int = 4860


@dataclass
class ModelConfig:
    """模型架構設定，僅描述模型本身，不包含資料切分與訓練流程設定。

    `lookback` 會在執行時依資料集中最長的動態樣本長度決定實際值。
    """

    # 每檔股票的輸入特徵維度數量。
    stock_feature_dim: int = 4

    # 市場整體特徵的輸入維度數量。
    market_feature_dim: int = 3

    # 股票時間序列編碼器的隱藏維度。
    stock_temporal_dim: int = 128

    # 市場時間序列編碼器的隱藏維度。
    market_temporal_dim: int = 64

    # 橫截面特徵整合層的隱藏維度。
    cross_sectional_dim: int = 128

    # 股票 ID embedding 的維度大小。
    stock_id_embedding_dim: int = 64

    # attention 機制使用的 head 數量。
    attention_heads: int = 8

    # dropout 比例，用來降低 overfitting 風險。
    dropout: float = 0.0

    def as_dict(self) -> dict:
        """將模型設定轉成 dict，方便記錄或輸出。"""
        return asdict(self)


@dataclass
class TrainConfig:
    """訓練流程設定，不包含資料規模設定，股票數量由 DataConfig 管理。"""

    # 隨機種子，確保訓練結果可重現。
    seed: int = 42

    # optimizer 使用的學習率。
    learning_rate: float = 5e-3

    # 每次訓練迭代使用的 batch 大小。
    batch_size: int = 1

    # 最大訓練 epoch 數量。
    num_epochs: int = 300

    # 權重衰減係數，用於 regularization。
    weight_decay: float = 1e-5

    # gradient clipping 的最大 norm，避免梯度爆炸。
    grad_clip_norm: float = 1.0

    # early stopping 容忍幾個 epoch 沒進步後停止訓練。
    early_stopping_patience: int = 100

    # diagnostic 模式中每次分析使用的步數或樣本步長設定。
    diagnostic_steps: int = 1

    # 訓練使用的 loss 名稱，例如 "return", "sharpe", "dsr", "sortino", "mdd", "cvar"。
    loss_name: str = "dsr"

    # 執行模式，通常為 "train" 或 "diagnostic"。
    mode: str = "train"

    # 執行裝置設定，例如 "cpu"、"cuda" 或 "auto"。
    device: str = "auto"

    # train mode 下最佳模型 checkpoint 的檔名。
    @property
    def train_best_checkpoint_name(self) -> str:
        return f"train_best_{self.loss_name}.pt"

    # train mode 下最後一個 epoch 模型 checkpoint 的檔名。
    @property
    def train_last_checkpoint_name(self) -> str:
        return f"train_last_{self.loss_name}.pt"

    # diagnostic mode 使用或輸出的 checkpoint 檔名。
    @property
    def checkpoint_name(self) -> str:
        return f"diagnostic_last_{self.loss_name}.pt"


@dataclass
class DiagnosticConfig:
    """診斷與圖表輸出相關設定。"""

    # allocation 圖表中保留前幾大群組，其餘合併為 Others。
    allocation_group_top_n: int = 7

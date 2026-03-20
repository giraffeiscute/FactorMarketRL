"""portfolio_attention 專案的設定輔助工具。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


def project_root() -> Path:
    # 取得目前檔案往上第 2 層的資料夾，作為專案根目錄
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    # 專案根目錄的上一層，作為整個 repo 根目錄
    return project_root().parent


def default_data_path() -> Path:
    # 預設資料集路徑
    return repo_root() / "toy_ff_generator" / "outputs" / "bull_4860_81_panel_long.csv"


@dataclass
class PathsConfig:
    """管理專案相關路徑設定。"""

    # 專案目錄，預設為 project_root()
    project_dir: Path = field(default_factory=project_root)

    # repo 目錄，預設為 repo_root()
    repo_dir: Path = field(default_factory=repo_root)

    # 輸出根目錄；若未指定，會自動使用 project_dir / "outputs"
    output_root: Path | None = None

    @property
    def outputs_dir(self) -> Path:
        # 所有輸出資料的主資料夾
        return self.output_root or self.project_dir / "outputs"

    @property
    def checkpoints_dir(self) -> Path:
        # 模型 checkpoint 存放資料夾
        return self.outputs_dir / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        # 指標結果存放資料夾
        return self.outputs_dir / "metrics"

    @property
    def logs_dir(self) -> Path:
        # 訓練或執行 log 存放資料夾
        return self.outputs_dir / "logs"

    @property
    def predictions_dir(self) -> Path:
        # 預測結果存放資料夾
        return self.outputs_dir / "predictions"


@dataclass
class DataConfig:
    """資料集相關設定。"""

    # 資料 CSV 路徑，預設使用 default_data_path()
    csv_path: Path = field(default_factory=default_data_path)

    # 回看視窗長度，例如用前 60 天/期資料作為輸入
    lookback: int = 60

    # 訓練集比例
    train_ratio: float = 0.75

    # 分析進場日；若為 None，會依 lookback 自動推算
    analysis_entry_day: int | None = None

    # 分析出場日；可選
    analysis_exit_day: int | None = None

    # 最多使用幾檔股票；None 表示不限制
    max_stocks: int | None = None

    def resolved_entry_day(self) -> int:
        # 若未手動指定 analysis_entry_day，
        # 則預設用 lookback + 1 作為進場日
        return self.analysis_entry_day or (self.lookback + 1)


@dataclass
class ModelConfig:
    """投資組合 attention 模型的維度設定。"""

    # 股票數量；若尚未確定可先為 None
    num_stocks: int | None = None

    # 每檔股票的特徵維度
    stock_feature_dim: int = 4

    # 市場整體特徵維度
    market_feature_dim: int = 3

    # 時序回看長度
    lookback: int = 60

    # 股票時間序列編碼後的維度
    stock_temporal_dim: int = 16

    # 市場時間序列編碼後的維度
    market_temporal_dim: int = 16

    # 橫截面（cross-sectional）特徵維度
    cross_sectional_dim: int = 16

    # 股票 ID embedding 維度
    stock_id_embedding_dim: int = 8

    # attention head 數量
    attention_heads: int = 1

    # dropout 比例
    dropout: float = 0.0

    def as_dict(self) -> dict:
        # 將 dataclass 轉成 dict，方便輸出、記錄或存檔
        return asdict(self)


@dataclass
class TrainConfig:
    """訓練與診斷執行相關設定。"""

    # 隨機種子，確保結果可重現
    seed: int = 7

    # 學習率
    learning_rate: float = 1e-3

    # 診斷步數
    diagnostic_steps: int = 1

    # 使用的 loss 名稱
    loss_name: str = "return"

    # 執行模式，例如 diagnostic / train
    mode: str = "diagnostic"

    # 使用裝置，auto 表示自動判斷 CPU/GPU
    device: str = "auto"

    # checkpoint 檔名
    checkpoint_name: str = "diagnostic_last.pt"

    # 訓練時最多使用幾檔股票；None 表示不限制
    max_stocks: int | None = None
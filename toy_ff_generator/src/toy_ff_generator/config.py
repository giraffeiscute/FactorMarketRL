from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

# 專案根目錄，從目前檔案位置往上回推兩層取得。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 市場狀態的固定順序：熊市、中性、牛市。
STATE_ORDER = (-1, 0, 1)

# 市場狀態數值與名稱的對應表。
STATE_NAME_MAP = {-1: "bear", 0: "neutral", 1: "bull"}

# mu 類別標籤，每一維都只分 low / mid / high 三類。
MU_CLASS_LABELS = ("low", "mid", "high")

# mu 的三個 characteristic 維度名稱。
MU_AXES = ("characteristic_1", "characteristic_2", "characteristic_3")

# alpha / epsilon 使用的群組標籤。
PROFILE_GROUP_LABELS = ("mid", "low", "high")


def _default_mu_class_centers() -> dict[str, float]:
    """回傳三種固定 mu_i 類別對應的數值中心。"""

    return {
        "low": -0.5,
        "mid": 0.0,
        "high": 0.5,
    }


def _default_mu_class_triplets(n: int) -> list[tuple[str, str, str]]:
    """以固定順序循環建立 27 種 mu_i 三維類別組合。"""

    # 針對三個 characteristic 維度，產生 low / mid / high 的所有笛卡兒積組合。
    all_triplets = list(product(MU_CLASS_LABELS, repeat=len(MU_AXES)))

    # 若股票數量超過 27，則依固定順序循環重複使用這些 triplets。
    return [all_triplets[idx % len(all_triplets)] for idx in range(n)]


def _default_stock_profiles(
    n: int,
) -> list[tuple[tuple[str, str, str], str, str]]:
    """建立依固定排序排列的股票 profile 區塊，每個 profile 代表一組 mu / alpha / epsilon 類型。"""

    # 先建立 27 種固定的 mu triplets。
    mu_triplets = _default_mu_class_triplets(len(MU_CLASS_LABELS) ** len(MU_AXES))

    # 建立完整 243 種 base profiles：
    # 27 種 mu triplets × 3 種 alpha_group × 3 種 epsilon_group。
    all_profiles = [
        (mu_triplet, alpha_group, epsilon_group)
        for alpha_group in PROFILE_GROUP_LABELS
        for epsilon_group in PROFILE_GROUP_LABELS
        for mu_triplet in mu_triplets
    ]

    # profile 總數，理論上應為 243。
    base_profile_count = len(all_profiles)

    # 每個 profile 至少要分配幾檔股票。
    base_block_size = n // base_profile_count

    # 前 remainder 個 profile 會再多分配 1 檔股票。
    remainder = n % base_profile_count

    stock_profiles: list[tuple[tuple[str, str, str], str, str]] = []

    # 依固定排序把 profile 轉成連續的股票區塊。
    for idx, profile in enumerate(all_profiles):
        block_size = base_block_size + (1 if idx < remainder else 0)
        stock_profiles.extend([profile] * block_size)

    return stock_profiles


def _default_per_stock_alpha_epsilon_groups(n: int) -> dict[str, list[str]]:
    """根據固定 stock profiles 建立每檔股票的 alpha 與 epsilon 群組指派。"""

    # 先取得每檔股票對應的 profile。
    stock_profiles = _default_stock_profiles(n)

    return {
        # 取出每檔股票對應的 alpha_group。
        "per_stock_alpha_groups": [alpha_group for _, alpha_group, _ in stock_profiles],
        # 取出每檔股票對應的 epsilon_group。
        "per_stock_epsilon_groups": [epsilon_group for _, _, epsilon_group in stock_profiles],
    }


def _triplets_to_mu_vectors(
    triplets: list[tuple[str, str, str]],
    class_centers: dict[str, float],
) -> list[list[float]]:
    """將固定的 mu_i 類別三元組轉成每檔股票對應的三維數值向量。"""

    return [
        [class_centers[class_label] for class_label in triplet]
        for triplet in triplets
    ]


def _default_per_stock_latent_state_params(n: int) -> dict[str, list[list[float]]]:
    """建立每檔股票的 latent characteristic state 參數，並保留固定的 mu_i。"""

    # 取得 mu 類別對應的數值中心。
    class_centers = _default_mu_class_centers()

    # 從每檔股票的 profile 中抽出對應的 mu triplet。
    mu_class_triplets = [mu_triplet for mu_triplet, _, _ in _default_stock_profiles(n)]

    # 將 mu triplets 轉成固定的三維 mu_i 數值向量。
    mu_i = _triplets_to_mu_vectors(mu_class_triplets, class_centers)

    return {
        # 每檔股票三維 latent state 的自迴歸係數。
        "Omega_i": [[0.65, 0.65, 0.65] for _ in range(n)],

        # 每檔股票固定不變的三維 mu_i。
        "mu_i": mu_i,

        # 市場狀態 S_t 對 latent state 的影響係數。
        "lambda_i": [[0.08, 0.05, 0.05] for _ in range(n)],

        # latent state 雜訊項的波動尺度。
        "sigma_Z_i": [[0.06, 0.06, 0.06] for _ in range(n)],

        # 初始 latent state，這裡預設直接設為對應的 mu_i。
        "Z0_i": [list(mu_vector) for mu_vector in mu_i],
    }


def _default_per_stock_initial_prices(n: int) -> list[float]:
    """建立每檔股票的預設初始價格序列。"""

    # 以固定間距建立可重現的初始價格。
    return [100.0 + 2.5 * idx for idx in range(n)]


def build_default_config() -> dict[str, Any]:
    """建立整個專案的預設模擬設定。"""

    # 預設股票數量。
    N = 4860

    # 預設模擬期數。
    T = 81

    return {
        "simulation_setup": {
            # 股票總數。
            "N": N,
            # 模擬總期數。
            "T": T,
            # 隨機種子，確保結果可重現。
            "random_seed": 42,
        },
        "market_state_setup": {
            # 若為 None，表示依轉移矩陣自動產生市場狀態序列。
            "state_sequence": None,
            # 初始市場狀態，0 代表 neutral。
            "initial_state": 1,
            # 市場狀態轉移矩陣，這裡預設為不轉移。
            "transition_matrix": [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
        },
        "factor_vector_ar_setup": {
            # 初始因子向量。
            "X0": [0.0, 0.0, 0.0],

            # 因子向量的 AR 係數矩陣。
            "Phi": [
                [0.6, 0.05, 0.02],
                [0.04, 0.55, 0.03],
                [0.02, 0.04, 0.25],
            ],

            # 不同市場狀態下的因子均值向量。
            "mu_bear": [-0.003, -0.003, 0.005],
            "mu_neutral": [0.0, 0.0, 0.0],
            "mu_bull": [0.002, 0.002, -0.0015],

            # 熊市下的因子共變異數矩陣。
            "Sigma_X_bear": [
                [0.00040, 0.000035, 0.000020],
                [0.000035, 0.000400, 0.000018],
                [0.000020, 0.000018, 0.000080],
            ],

            # 中性市場下的因子共變異數矩陣。
            "Sigma_X_neutral": [
                [0.00012, 0.000015, 0.000008],
                [0.000015, 0.000065, 0.000010],
                [0.000008, 0.000010, 0.000055],
            ],

            # 牛市下的因子共變異數矩陣。
            "Sigma_X_bull": [
                [0.00009, 0.000018, 0.000005],
                [0.000018, 0.000055, 0.000006],
                [0.000005, 0.000006, 0.000045],
            ],
        },
        "mu_class_setup": {
            # mu 類別中心值設定。
            "class_centers": _default_mu_class_centers(),
        },
        "latent_characteristic_setup": {
            # 是否使用所有股票共用同一組 latent state 參數。
            "use_shared_latent_state_params": False,

            "shared_params": {
                # 共用模式下的 Omega。
                "Omega": [0.65, 0.65, 0.65],
                # 共用模式下的 mu_Z。
                "mu_Z": [0.0, 0.0, 0.0],
                # 共用模式下的 lambda_Z。
                "lambda_Z": [0.05, 0.03, 0.03],
                # 共用模式下的 sigma_Z。
                "sigma_Z": [0.05, 0.05, 0.05],
                # 共用模式下的初始 latent state。
                "Z0": [0.0, 0.0, 0.0],
            },

            # 每檔股票各自獨立的 latent state 參數。
            "per_stock_params": _default_per_stock_latent_state_params(N),
        },
        "exposure_setup": {
            # latent characteristics 映射到 exposure 的線性轉換矩陣。
            "A": [
                [0.05, 0.0, 0.0],
                [0.0, 0.4, 0.0],
                [0.0, 0.0, 0.4],
            ],

            # 線性轉換的常數偏移項。
            "b": [1, 0.4, 0.0],
        },
        "alpha_epsilon_mode_setup": {
            # 舊版 shared fallback 的 alpha_group。
            "alpha_group": "mid",

            # 舊版 shared fallback 的 epsilon_group。
            "epsilon_group": "mid",

            # alpha 各群組對應的數值設定。
            "alpha_levels": {
                "low": -0.0001,
                "mid": 0.0002,
                "high": 0.0003,
            },

            # epsilon 各群組對應的數值設定。
            "epsilon_levels": {
                "low": 0.005,
                "mid": 0.01,
                "high": 0.02,
            },

            # 每檔股票固定的 alpha / epsilon 群組配置。
            **_default_per_stock_alpha_epsilon_groups(N),
        },
        "clipping_price_setup": {
            # 單期最大漲幅限制。
            "limit_up": 0.10,

            # 單期最大跌幅限制。
            "limit_down": -0.10,

            # 是否所有股票共用同一個初始價格。
            "shared_init_price": True,

            # 共用初始價格。
            "initial_price": 100.0,

            # 每檔股票各自的初始價格。
            "per_stock_initial_price": _default_per_stock_initial_prices(N),
        },
        "output_setup": {
            # 輸出資料夾位置。
            "output_dir": str(PROJECT_ROOT / "outputs"),
        },
    }
import pandas as pd
import numpy as np
import os
import itertools
from pathlib import Path
import ast

# ==========================================
# 參數設定 (在此修改你要處理的資料集)
# ==========================================
STATE = "neutral"  # 可選: "bull", "bear", "neutral"
N = 4860
T = 200
# ==========================================

def standardize_mu(val):
    """將 mu 值轉換為標準格式字串，例如 '(0.5, 0.0, -0.5)'。"""
    try:
        if isinstance(val, str):
            # 嘗試解析 tuple 字串
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (tuple, list)):
                return str(tuple(float(x) for x in parsed))
        elif isinstance(val, (tuple, list)):
            return str(tuple(float(x) for x in val))
        return str(val)
    except:
        return str(val)

def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    """處理帶有百分比符號的數值欄位。"""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="raise")

    # 轉為字串並去除空格
    normalized = series.astype(str).str.strip()
    
    # 檢查是否有百分比符號
    percent_mask = normalized.str.endswith("%")
    if percent_mask.any():
        normalized = normalized.str.replace("%", "", regex=False)
        numeric = pd.to_numeric(normalized, errors="coerce")
        # 如果是百分比，除以 100
        return numeric / 100.0
    return pd.to_numeric(normalized, errors="coerce")

def main():
    # 1. 設定路徑
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "toy_ff_generator" / "outputs"
    input_file = output_dir / f"{STATE}_{N}_{T}_panel_long.csv"
    output_file = output_dir / f"{STATE}_{N}_{T}_combo_stats.csv"
    
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        return

    print(f"--- Processing: {STATE} | N={N} | T={T} ---")
    print(f"Reading: {input_file}")
    
    # 2. 讀取與數值轉換
    df = pd.read_csv(input_file)
    print(f"Original records: {len(df)}")

    # 找出報酬欄位
    return_priority = ["return"]
    return_col = next((c for c in return_priority if c in df.columns), None)
    if not return_col:
        raise ValueError(f"Could not find return column. Available: {df.columns.tolist()}")
    
    # 轉換數值型態 (處理百分比符號)
    print(f"Cleaning columns: '{return_col}', 'alpha', 'epsilon_variance'...")
    df[return_col] = _coerce_numeric_series(df[return_col])
    
    # epsilon 可能被稱為 epsilon_variance
    epsilon_col = "epsilon_variance" if "epsilon_variance" in df.columns else "epsilon_variance"
    df["alpha_numeric"] = _coerce_numeric_series(df["alpha"])
    df["epsilon_numeric"] = _coerce_numeric_series(df[epsilon_col])
    
    # 3. 預處理分組欄位
    df["mu_std"] = df["mu"].apply(standardize_mu)
    
    # 4. 計算統計量
    stats = df.groupby(["mu_std", "alpha_numeric", "epsilon_numeric"])[return_col].agg(
        mean_return="mean",
        std_return="std",
        count="count"
    ).reset_index()

    # 5. 建立 243 種組合的模版
    mu_elements = [-0.5, 0.0, 0.5]
    mu_combos = [str(tuple(float(x) for x in p)) for p in itertools.product(mu_elements, repeat=3)]
    
    # 取出 alpha 和 epsilon 的唯一值
    alpha_vals = sorted(df["alpha_numeric"].unique().tolist())
    epsilon_vals = sorted(df["epsilon_numeric"].unique().tolist())
    
    # 建立完整網格
    full_grid = pd.DataFrame(
        list(itertools.product(mu_combos, alpha_vals, epsilon_vals)),
        columns=["mu_std", "alpha", "epsilon"]
    )

    # 6. 合併與補齊
    final_df = pd.merge(
        full_grid, 
        stats.rename(columns={"alpha_numeric": "alpha", "epsilon_numeric": "epsilon"}), 
        on=["mu_std", "alpha", "epsilon"], 
        how="left"
    )
    
    # 處理缺失值
    final_df["count"] = final_df["count"].fillna(0).astype(int)
    final_df = final_df.rename(columns={"mu_std": "mu"})

    # 6.5 計算 Spearman 相關係數 (整體資料)
    # 提取 mu 的第一個元素 (mu[0]) 用於相關性計算
    mu0_series = final_df["mu"].apply(lambda x: ast.literal_eval(x)[0])

    # 計算各項相關係數摘要
    corrs = {
        "spearman_alpha_mean": final_df["alpha"].rank().corr(final_df["mean_return"].rank()),
        "spearman_epsilon_std": final_df["epsilon"].rank().corr(final_df["std_return"].rank()),
        "spearman_epsilon_mean": final_df["epsilon"].rank().corr(final_df["mean_return"].rank()),
        "spearman_mu0_mean": mu0_series.rank().corr(final_df["mean_return"].rank()),
        "spearman_mu0_std": mu0_series.rank().corr(final_df["std_return"].rank())
    }
    
    # ==========================================
    # 排序邏輯：按 mean_return 由大到小排序
    # ==========================================
    final_df = final_df.sort_values(by="mean_return", ascending=False).reset_index(drop=True)

    # 存入欄位，只在第一行保留數值 (避免重複資訊)
    for col, val in corrs.items():
        final_df[col] = np.nan
        final_df.at[0, col] = val
    # ==========================================

    # 7. 輸出結果
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(output_file, index=False)
    
    print("\nSummary Results (Sorted by mean_return DESC):")
    print(f"Total unique combinations expected: {len(full_grid)}")
    print(f"Combinations found in data: {len(stats)}")
    print(f"Combinations added (count=0): {len(final_df) - len(stats)}")
    print(f"Final output rows: {len(final_df)}")
    print(f"Top 5 performers:\n{final_df.head(5)[['mu', 'alpha', 'epsilon', 'mean_return']]}")
    print(f"Output saved to: {output_file}")
    
    # 顯示相關係數摘要
    print("\nSpearman Rank Correlations (Full Dataset):")
    print(f"  - alpha vs mean_return:   {corrs['spearman_alpha_mean']:.4f}")
    print(f"  - epsilon vs std_return:  {corrs['spearman_epsilon_std']:.4f}")
    print(f"  - epsilon vs mean_return: {corrs['spearman_epsilon_mean']:.4f}")
    print(f"  - mu[0] vs mean_return:   {corrs['spearman_mu0_mean']:.4f}")
    print(f"  - mu[0] vs std_return:    {corrs['spearman_mu0_std']:.4f}")

if __name__ == "__main__":
    main()

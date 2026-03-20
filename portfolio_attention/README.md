# portfolio_attention

`portfolio_attention` 已重建為可執行的最小完整專案，採用 `src / outputs / tests` 結構，定位為 `diagnostic / executable baseline`，不是正式 OOS backtest baseline。

## 目錄結構

```text
portfolio_attention/
  README.md
  pyproject.toml
  src/portfolio_attention/
  outputs/
    checkpoints/
    metrics/
    logs/
    predictions/
  tests/
```

- source code 在 `src/portfolio_attention/`
- `outputs/` 只存模型產物，不存 raw CSV
- tests 在 `tests/`

## 原始資料

- 預設資料檔：`./toy_ff_generator/outputs/bull_4860_81_panel_long.csv`
- 檔名規則：`{prefix_}N_T_panel_long.csv`
- 由檔名解析本資料得到：
  - `N = 4860`
  - `T = 81`
- `dataset.py` 會再用 CSV 驗證：
  - `unique(stock_id) == 4860`
  - `unique(t) == 81`

## 固定 schema

模型只讀取以下固定欄位，不做欄位名猜測：

- `stock_id`
- `t`
- `characteristic_1`
- `characteristic_2`
- `characteristic_3`
- `MKT`
- `SMB`
- `HML`
- `price`

若 CSV 仍有額外欄位，程式會忽略它們，但不會把它們視為模型輸入。

## 模型流程

文字流程如下：

1. 每支股票的輸入固定為 `[characteristic_1, characteristic_2, characteristic_3, price]`
2. 對每支股票沿時間維做 stock temporal encoder
3. 時間位置編碼使用加法：`x_t = w_t + p_t`
4. 由時間編碼結果做 temporal pooling，得到每支股票 summary `z_i`
5. 對市場 FF3 序列 `[MKT, SMB, HML]` 單獨做 market temporal encoder，得到 market summary `m`
6. 融合方式保留為 `z_i_prime = [z_i ; m]`
7. 投影到 cross-sectional hidden space
8. 股票 identity position 使用拼接：`x_{s,t} = [z_{s,t}; e_s]`
9. 只在股票維度做 stock attention，cash 不會成為 attention token
10. 由股票 token 產生 stock logits，並由 pooled stock representation 產生 cash logit
11. `stock logits + cash logit` 一起做 softmax，得到最終配置

## 時間切分與 sample 定義

- 時間切分採前 `3/4` 為 train、後 `1/4` 為 test
- 對 `T = 81`：
  - `train = 60`
  - `test = 21`
- 固定 sample 定義：
  - `lookback = 60`
  - 第 `61` 天決定配置
  - 持有到第 `80` 天
  - `R_i = price_exit / price_entry - 1`
  - `entry = day 61`
  - `exit = day 80`

在這個定義下：

- 合法 training windows = `0`
- 合法 test windows = `0`

原因：

- train split 只有前 60 天，無法同時容納 `60` 天 lookback 與後續持有期
- test split 只有後 21 天，無法提供 `60` 天 lookback

因此本專案不假裝存在正常 in-sample / out-of-sample rolling windows，只保留 1 個 `cross-boundary analysis window`：

- lookback：days `1..60`
- entry：day `61`
- exit：day `80`

這個 window 只用於：

- pipeline 可執行性驗證
- tensor shape 檢查
- forward / loss / allocation 診斷
- smoke test
- baseline diagnostic output

它不是正式訓練 sample，也不是正式 OOS test sample。

## CPU / GPU 執行

裝套件：

```bash
cd portfolio_attention
pip install -e .
```

CPU 本機 diagnostic：

```bash
cd portfolio_attention
python -m portfolio_attention.train --mode diagnostic
python -m portfolio_attention.evaluate --mode diagnostic
```

若只想做更快的本機驗證，可暫時限制股票數量：

```bash
cd portfolio_attention
python -m portfolio_attention.train --mode diagnostic --max-stocks 128
```

GPU 伺服器：

```bash
cd portfolio_attention
python -m portfolio_attention.train --mode diagnostic --device cuda
python -m portfolio_attention.evaluate --mode diagnostic --device cuda
```

若 `--device auto`，程式會自動選擇 CUDA，否則退回 CPU。載入 checkpoint 時會使用相容的 `map_location`。

## 測試

```bash
cd portfolio_attention
pytest
```

## 評估輸出

`evaluate.py` 會輸出並保存：

- portfolio return
- average cash weight
- top-k stock weights
- sharpe-like metric

所有結果都會標示 `diagnostic_only = true`，因為目前只有單一 cross-boundary analysis window。

## 目前 baseline 的限制

- 目前資料條件下沒有合法 train/test windows，無法聲稱正式 OOS backtest
- `train.py` 只執行 diagnostic optimization / forward validation，不會偽造成正常訓練
- 單一 analysis window 對 sharpe-style loss 極不穩定，因此程式在 batch 太小時會 fallback 到 mean-return style loss 並發出 warning

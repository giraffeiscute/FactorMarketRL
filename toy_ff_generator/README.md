# Toy FF Generator

這個專案是一個透明、可檢查、可解釋的 toy data generating process（DGP），用來模擬股票報酬與價格。它的用途是做檢查、除錯與實驗，不是實盤交易策略。

# 模型生成流程

這個 toy generator 的核心流程如下。

## Step 1：設定市場狀態序列

先定義整段模擬期間的市場狀態：

$$
S_1, \ldots, S_T,\qquad S_t \in \{-1,0,1\}
$$

其中：

- $S_t=-1$：bear market
- $S_t=0$：neutral market
- $S_t=1$：bull market

目前支援兩種方式：

- 直接手動指定整段 state sequence
- 由初始狀態與 Markov transition matrix 生成 state sequence

---

## Step 2：生成 FF 三因子

這一版不再把三個因子分別視為獨立 AR(1)，而是把它們視為一個 3 維向量：

$$
X_t = [MKT_t, SMB_t, HML_t]^T
$$

並使用向量 AR(1)：

$$
X_t=\Phi X_{t-1}+\Delta S_t+u_t
$$

其中：

$$
u_t \sim N(0,\Sigma_X(S_t))
$$

說明如下：

- $\Phi$ 是 $3\times 3$ 矩陣
- $\Delta$ 是長度為 3 的向量
- $S_t$ 會影響因子系統的均值位移
- regime 也會影響 factor innovation covariance
- bear / neutral / bull 可分別使用不同的 covariance matrix

也就是：

$$
\Sigma_X^{bear},\qquad \Sigma_X^{neutral},\qquad \Sigma_X^{bull}
$$

分別對應：

- bear state 時的 $\Sigma_X(S_t)$
- neutral state 時的 $\Sigma_X(S_t)$
- bull state 時的 $\Sigma_X(S_t)$

對應關係為：

$$
X_{1,t}=MKT_t
$$

$$
X_{2,t}=SMB_t
$$

$$
X_{3,t}=HML_t
$$

---

## Step 3：生成個股 characteristic

這一版 characteristic 不再是每一期獨立抽樣，而是改成具有慣性（persistence / inertia）的動態過程。

對每支股票 $i$，定義：

$$
C_{i,t}=\Omega_i C_{i,t-1}+\mu_i+\Lambda_i S_t+\xi_{i,t}
$$

其中：

$$
\xi_{i,t}\sim N(0,\sigma_{C,i}^2)
$$

這代表：

- characteristic 不再是 i.i.d. across time
- $C_{i,t}$ 會依賴前一期的 $C_{i,t-1}$
- $\Omega_i$ 控制 characteristic 的 persistence
- 初值 $C_{i,0}$ 可手動設定

目前支援兩種參數模式：

- shared params：
  所有股票共用同一組 $\Omega,\mu_C,\Lambda_C,\sigma_C,C_0$
- per-stock params：
  每支股票各自擁有 $\Omega_i,\mu_i,\Lambda_i,\sigma_{C,i},C_{i,0}$

---

## Step 4：由 characteristic 生成因子曝險

對每支股票、每個時間點、每個因子曝險，定義：

$$
\beta_{i,t,k}=g_k(C_{i,t})
$$

目前仍採最簡單的線性單調形式：

$$
\beta_{i,t,k}=a_k C_{i,t}+b_k
$$

因此三個曝險分別為：

- `beta_mkt`
- `beta_smb`
- `beta_hml`

這一版先不引入更複雜的非線性 $g_k(\cdot)$。

---

## Step 5：生成個股固定效果與噪音

對每支股票 $i$，生成固定效果：

$$
\alpha_i \sim N(\mu_\alpha,\sigma_\alpha^2)
$$

另外生成 idiosyncratic noise：

$$
\varepsilon_{i,t}\sim N(0,\sigma_{\varepsilon,i}^2)
$$

其中 epsilon 目前支援：

- shared $\sigma_\varepsilon$
- per-stock $\sigma_{\varepsilon,i}$

---

## Step 6：生成個股報酬

本版維持當期對齊（contemporaneous alignment）：

$$
r_{i,t}=\alpha_i+\beta_{i,t,1}MKT_t+\beta_{i,t,2}SMB_t+\beta_{i,t,3}HML_t+\varepsilon_{i,t}
$$

也就是：

- factor 的 $t$ 使用 $X_t$
- characteristic 的 $t$ 使用 $C_{i,t}$
- beta 的 $t$ 使用 $\beta_{i,t}$
- epsilon 的 $t$ 使用 $\varepsilon_{i,t}$

這次更新只改：

- factor dynamics
- characteristic dynamics

目前不引入 return lag 結構。

在報酬生成後，程式還會進一步：

1. 對 raw return 做 clipping
2. 用 clipping 後的 return 遞推價格
3. 輸出 wide / long 兩種格式資料

---

# 需要手動調整的輸入參數

這些參數都集中放在主入口檔案 `src/toy_ff_generator/main.py`，方便直接手改。

## (A) 基本維度與市場狀態

- $N$：股票數量
- $T$：時間點數量
- random seed
- 市場狀態序列 $\{S_t\}_{t=1}^T$ 或其生成方式
  - 可手動指定 `state_sequence`
  - 或指定 `initial_state` 與 Markov transition matrix

## (B) characteristic $C_{i,t}$ 的參數

新版 characteristic 參數改成動態遞迴版本。

shared 模式下，主要參數為：

- $\Omega$
- $\mu_C$
- $\Lambda_C$
- $\sigma_C$
- $C_0$

per-stock 模式下，主要參數為：

- $\Omega_i$
- $\mu_i$
- $\Lambda_i$
- $\sigma_{C,i}$
- $C_{i,0}$

也就是現在不再只是一組靜態的 $\mu_C,\lambda_C,\sigma_C$ 抽樣設定，而是完整描述 characteristic 慣性過程的參數。

## (C) beta 函數 $g_k(\cdot)$ 的參數

目前仍使用線性形式：

$$
\beta_{i,t,k}=a_k C_{i,t}+b_k
$$

對應需要設定：

- $a_1,a_2,a_3$
- $b_1,b_2,b_3$

分別對應：

- `beta_mkt`
- `beta_smb`
- `beta_hml`

## (D) FF 三因子向量 AR 的參數

舊版逐因子獨立設定的

- $\phi_k$
- $\delta_k$
- $\sigma_{X,k}$
- $\rho_{X,k}$

已改成向量系統參數：

- $X_0$
- $\Phi$
- $\Delta$
- $\Sigma_X^{bear}$
- $\Sigma_X^{neutral}$
- $\Sigma_X^{bull}$

其中：

- $X_0$ 是長度 3 的初始向量
- $\Phi$ 是 $3 \times 3$ matrix
- $\Delta$ 是長度 3 向量
- 三個 covariance matrix 分別對應 bear / neutral / bull regime

## (E) 個股固定效果與噪音參數

- $\mu_\alpha,\sigma_\alpha$：控制 $\alpha_i$
- shared $\sigma_\varepsilon$ 或 per-stock $\sigma_{\varepsilon,i}$
- return clipping bounds
- initial price

---

# 輸出 / 中間產物

這個專案會輸出最終報酬資料，也會保留可檢查的中間結果。

## 最終輸出

股票報酬資料為：

$$
\{r_{i,t}\}_{i=1,\ldots,N;\;t=1,\ldots,T}
$$

並整理成 $N\times T$ 的 wide format。

另外價格會由 clipping 後的 return 遞推生成：

$$
P_{i,t}=P_{i,t-1}(1+r_{i,t})
$$

## 中間產物

除了最終輸出外，也會保留下列可檢查的白箱資料：

- 市場狀態序列

$$
\{S_t\}_{t=1}^T
$$

- FF 三因子向量序列

$$
\{X_t\}_{t=1}^T
$$

- 個股 characteristic

$$
\{C_{i,t}\}
$$

- 個股因子曝險

$$
\{\beta_{i,t,1},\beta_{i,t,2},\beta_{i,t,3}\}
$$

- 個股固定效果

$$
\{\alpha_i\}
$$

- 個股噪音

$$
\{\varepsilon_{i,t}\}
$$

對應資料表欄位格式為：

- `factor_df`: `[t, MKT, SMB, HML]`
- `characteristic_df`: `[stock_id, t, C]`
- `beta_df`: `[stock_id, t, beta_mkt, beta_smb, beta_hml]`
- `alpha_df`: `[stock_id, alpha]`
- `epsilon_df`: `[stock_id, t, epsilon]`
- `panel_long_df` 至少包含：
  `[stock_id, t, C, alpha, beta_mkt, beta_smb, beta_hml, MKT, SMB, HML, epsilon, raw_return, return, price]`

---

# 11. 核心白箱數學總結

為了方便快速回顧，核心模型可整理如下：

$$
S_t\in\{-1,0,1\}
$$

$$
X_t=[MKT_t,SMB_t,HML_t]^T
$$

$$
X_t=\Phi X_{t-1}+\Delta S_t+u_t,\qquad
u_t\sim N(0,\Sigma_X(S_t))
$$

$$
\Sigma_X(S_t)\in\left\{\Sigma_X^{bear},\Sigma_X^{neutral},\Sigma_X^{bull}\right\}
$$

$$
C_{i,t}=\Omega_i C_{i,t-1}+\mu_i+\Lambda_i S_t+\xi_{i,t},\qquad
\xi_{i,t}\sim N(0,\sigma_{C,i}^2)
$$

$$
\beta_{i,t,k}=a_k C_{i,t}+b_k,\qquad k=1,2,3
$$

$$
r_{i,t}=\alpha_i+\beta_{i,t,1}MKT_t+\beta_{i,t,2}SMB_t+\beta_{i,t,3}HML_t+\varepsilon_{i,t}
$$

$$
r_{i,t}^{obs}=\operatorname{clip}(r_{i,t},\text{limit\_down},\text{limit\_up})
$$

$$
P_{i,t}=P_{i,t-1}(1+r_{i,t}^{obs})
$$

## 安裝

在專案根目錄執行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 執行

目前專案統一由 `main.py` 啟動。

最直接的執行方式：

```bash
python -m toy_ff_generator.main
```

如果想直接手動調整參數，請修改：

```text
src/toy_ff_generator/main.py
```

你可以在 `build_default_config()` 中調整：

- `N`
- `T`
- `random_seed`
- `state_sequence` 或 `transition_matrix`
- factor vector AR 參數
- characteristic inertia 參數
- exposure 參數
- alpha / epsilon / clipping / price / output 參數

## 輸出格式

程式會在 `outputs/` 目錄下輸出 4 個檔案：

1. `returns.csv`
   - `N x T` 的 wide matrix
   - 列為 `stock_id`
   - 欄為 `t_0, t_1, ..., t_{T-1}`
   - 值為 clipping 後的 return

2. `prices.csv`
   - `N x T` 的 wide matrix
   - 列為 `stock_id`
   - 欄為 `t_0, t_1, ..., t_{T-1}`
   - 值為由 clipping 後 return 遞推得到的價格

3. `panel_long.csv`
   - long panel 格式
   - 欄位至少包含：
     `stock_id, t, C, alpha, beta_mkt, beta_smb, beta_hml, MKT, SMB, HML, epsilon, raw_return, return, price`

4. `metadata.json`
   - 儲存本次模擬所使用的主要設定參數

## 測試

在專案根目錄執行：

```bash
python -m pytest
```

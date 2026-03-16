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
X_t=\Phi X_{t-1}+\mu_X(S_t)+u_t
$$

其中：

$$
u_t \sim N(0,\Sigma_X(S_t))
$$

說明如下：

- $\Phi$ 是 $3\times 3$ 矩陣
- $\mu_X(S_t)$ 是由 regime 決定的長度 3 mean vector
- $S_t$ 是 scalar regime state，不是矩陣
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

這一版 characteristic 不再是單一 scalar，也不再是每一期獨立抽樣，而是改成具有慣性（persistence / inertia）的三維動態向量。

對每支股票 $i$，定義：

$$
C_{i,t}=
\begin{bmatrix}
C_{i,t}^{(1)}\\
C_{i,t}^{(2)}\\
C_{i,t}^{(3)}
\end{bmatrix}
$$

並採用逐維遞迴形式：

$$
C_{i,t}=\Omega_i \odot C_{i,t-1}+\mu_i+\Lambda_i S_t+\xi_{i,t}
$$

其中：

$$
\xi_{i,t}\sim N(0,\Sigma_{C,i})
$$

為了保持實作簡潔，現在採用 diagonal covariance 的版本，也就是三個 characteristic shock 彼此獨立：

$$
\Sigma_{C,i}=
\operatorname{diag}\left(
(\sigma_{C,i}^{(1)})^2,
(\sigma_{C,i}^{(2)})^2,
(\sigma_{C,i}^{(3)})^2
\right)
$$

因此也可以逐維寫成：

$$
C_{i,t}^{(d)}=\Omega_i^{(d)} C_{i,t-1}^{(d)}+\mu_i^{(d)}+\Lambda_i^{(d)} S_t+\xi_{i,t}^{(d)},
\qquad d\in\{1,2,3\}
$$

且

$$
\xi_{i,t}^{(d)} \sim N\left(0,(\sigma_{C,i}^{(d)})^2\right)
$$

這代表：

- characteristic 不再是 i.i.d. across time
- 每個 stock-time pair 都有一個 3 維 characteristic vector
- $C_{i,t}$ 會依賴前一期的 $C_{i,t-1}$
- $\Omega_i$ 控制各維 characteristic 的 persistence
- 初值 $C_{i,0}$ 可手動設定

目前支援兩種參數模式：

- shared params：
  所有股票共用同一組長度 3 的 $\Omega,\mu_C,\Lambda_C,\sigma_C,C_0$
- per-stock params：
  每支股票各自擁有 shape 為 $(3,)$ 的 $\Omega_i,\mu_i,\Lambda_i,\sigma_{C,i},C_{i,0}$

---

## Step 4：由 characteristic 生成因子曝險

對每支股票、每個時間點、每個因子曝險，定義：

$$
\beta_{i,t,k}=g_k(C_{i,t})
$$

目前仍採最簡單的線性形式，但因為 characteristic 已經是三維向量，所以改成：

$$
\beta_{i,t,k}=b_k+a_k^T C_{i,t}
$$

其中：

- $C_{i,t}$ 是長度 3 的 characteristic vector
- $a_k$ 是長度 3 的 loading vector
- $b_k$ 是 scalar

也就是三個曝險分別為：

$$
\beta_{mkt}=b_{mkt}+a_{mkt}^T C_{i,t}
$$

$$
\beta_{smb}=b_{smb}+a_{smb}^T C_{i,t}
$$

$$
\beta_{hml}=b_{hml}+a_{hml}^T C_{i,t}
$$

對應程式中的欄位名稱：

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
- beta 由單一 scalar characteristic 改成 characteristic vector 的線性映射

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

新版 characteristic 參數改成三維動態遞迴版本。

shared 模式下，主要參數為：

- $\Omega$：長度 3
- $\mu_C$：長度 3
- $\Lambda_C$：長度 3
- $\sigma_C$：長度 3
- $C_0$：長度 3

per-stock 模式下，主要參數為：

- $\Omega_i$：shape $(N,3)$
- $\mu_i$：shape $(N,3)$
- $\Lambda_i$：shape $(N,3)$
- $\sigma_{C,i}$：shape $(N,3)$
- $C_{i,0}$：shape $(N,3)$

也就是現在不再是一組單一 scalar 的 $\mu_C,\lambda_C,\sigma_C$ 抽樣設定，而是完整描述三維 characteristic 慣性過程的參數。

## (C) beta 函數 $g_k(\cdot)$ 的參數

目前使用向量線性形式：

$$
\beta_{i,t,k}=b_k+a_k^T C_{i,t}
$$

對應需要設定：

- `a_mkt`：長度 3
- `a_smb`：長度 3
- `a_hml`：長度 3
- `b_mkt`
- `b_smb`
- `b_hml`

分別對應：

- `beta_mkt`
- `beta_smb`
- `beta_hml`

## (D) FF 三因子向量 AR 的參數

舊版逐因子獨立設定的

- $\Phi$
- $\mu_X^{bear}$
- $\mu_X^{neutral}$
- $\mu_X^{bull}$

已改成向量系統參數：

- $X_0$
- $\Phi$
- $\mu_X^{bear}$
- $\mu_X^{neutral}$
- $\mu_X^{bull}$
- $\Sigma_X^{bear}$
- $\Sigma_X^{neutral}$
- $\Sigma_X^{bull}$

其中：

- $X_0$ 是長度 3 的初始向量
- $\Phi$ 是 $3 \times 3$ matrix
- $\mu_X^{bear}, \mu_X^{neutral}, \mu_X^{bull}$ 是三組 regime-specific 長度 3 mean vector
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
P_{i,t}=P_{i,t-1}(1+r_{i,t}^{obs})
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
- `characteristic_df`: `[stock_id, t, C1, C2, C3]`
- `beta_df`: `[stock_id, t, beta_mkt, beta_smb, beta_hml]`
- `alpha_df`: `[stock_id, alpha]`
- `epsilon_df`: `[stock_id, t, epsilon]`
- `panel_long_df` 至少包含：
  `[stock_id, t, C1, C2, C3, alpha, beta_mkt, beta_smb, beta_hml, MKT, SMB, HML, epsilon, raw_return, return, price]`

---

# 11. 核心白箱數學總結

為了方便快速回顧，核心模型可整理如下：

i 表示資產（或公司）索引， 

k 表示風險因子（factor）的索引，  對應 Fama–French 三因子：
$
(\mathrm{MKT},\ \mathrm{SMB},\ \mathrm{HML})
$


$$
S_t\in\{-1,0,1\}
$$

$$
\mathbf{X}_t =
\begin{bmatrix}
\mathrm{MKT}_t \\
\mathrm{SMB}_t \\
\mathrm{HML}_t
\end{bmatrix}
$$


$$
\mathbf{X}_t
=
\Phi \mathbf{X}_{t-1}
+
\mu_X(S_t)
+
\mathbf{u}_t,
\qquad
\mathbf{u}_t \sim \mathcal{N}\!\left(\mathbf{0},\,\Sigma_X(S_t)\right)
\qquad
\left(
\Phi \in \mathbb{R}^{3\times 3},\ 
\mu_X(S_t) \in \mathbb{R}^{3\times 1},\ 
\mathbf{u}_t \in \mathbb{R}^{3\times 1},\ 
\Sigma_X(S_t)\in\mathbb{R}^{3\times 3}
\right)
$$ 

$$
\mu_X(S_t)
\in
\left\{
\mu_X^{\mathrm{bear}},
\mu_X^{\mathrm{neutral}},
\mu_X^{\mathrm{bull}}
\right\}
\qquad
\left(
\mu_X^{\mathrm{bear}},
\mu_X^{\mathrm{neutral}},
\mu_X^{\mathrm{bull}}
\in \mathbb{R}^{3\times 1}
\right)
$$

$$
\Sigma_X(S_t)
\in
\left\{
\Sigma_X^{\mathrm{bear}},
\Sigma_X^{\mathrm{neutral}},
\Sigma_X^{\mathrm{bull}}
\right\}
\qquad
\left(
\Sigma_X^{\mathrm{bear}},
\Sigma_X^{\mathrm{neutral}},
\Sigma_X^{\mathrm{bull}}
\in \mathbb{R}^{3\times 3}
\right)
$$


$$
\mathbf{C}_{i,t}
=
\begin{bmatrix}
C_{i,t}^{(1)} \\
C_{i,t}^{(2)} \\
C_{i,t}^{(3)}
\end{bmatrix}
\qquad
\left(
\mathbf{C}_{i,t} \in \mathbb{R}^{3\times1}
\right)
$$

$$
\mathbf{C}_{i,t}
=
\Omega_i \mathbf{C}_{i,t-1}
+
\boldsymbol{\mu}_i
+
\boldsymbol{\lambda}_i S_t
+
\boldsymbol{\xi}_{i,t}
$$

$$
\left(
\Omega_i \in \mathbb{R}^{3\times3},\;
\boldsymbol{\mu}_i \in \mathbb{R}^{3\times1},\;
\boldsymbol{\lambda}_i \in \mathbb{R}^{3\times1},\;
\boldsymbol{\xi}_{i,t} \in \mathbb{R}^{3\times1}
\right)
$$

$$
\boldsymbol{\xi}_{i,t}
\sim
\mathcal{N}
\left(
\mathbf{0},
\Sigma_{C,i}
\right)
$$

$$
\Omega_i
=
\operatorname{diag}
\left(
\omega_i^{(1)},
\omega_i^{(2)},
\omega_i^{(3)}
\right)
$$

$$
\Sigma_{C,i}
=
\operatorname{diag}
\left(
(\sigma_{C,i}^{(1)})^2,
(\sigma_{C,i}^{(2)})^2,
(\sigma_{C,i}^{(3)})^2
\right)
\qquad
\left(
\Sigma_{C,i}\in\mathbb{R}^{3\times3}
\right)
$$

$$
\beta_{i,t,k}
=
b_k
+
\mathbf{a}_k^\top
\mathbf{C}_{i,t},
\qquad
k=1,2,3
\qquad
\left(
\mathbf{a}_k \in \mathbb{R}^{3\times1},\;
\beta_{i,t,k},\, b_k \in \mathbb{R}
\right)
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
- characteristic vector inertia 參數
- exposure loading vectors
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
     `stock_id, t, C1, C2, C3, alpha, beta_mkt, beta_smb, beta_hml, MKT, SMB, HML, epsilon, raw_return, return, price`

4. `metadata.json`
   - 儲存本次模擬所使用的主要設定參數

## 測試

在專案根目錄執行：

```bash
python -m pytest
```

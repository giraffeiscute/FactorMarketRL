# Portfolio Attention 模型邏輯文檔

## 1. 問題設定與時間記號

對於一個 sample，我們同時觀察 $N$ 支股票與一段市場因子序列。

### 1.1 股票與時間軸

* 股票索引：$i \in {1,\dots,N}$
* **回看長度**：$L = 60$
* **整段 sample 的總時間長度**：$S = 81$

這兩個量要明確分開：

* $L=60$：模型真正看到的歷史長度
* $S=81$：資料檔內這個 sample 的總時間長度（例如 $t_0,\dots,t_{80}$）

因此，模型的流程是：

* 用前 60 天資訊做決策
* 在下一個時點建倉
* 持有到後續指定平倉時點
* 再用該持有期報酬作為訓練訊號

---

## 2. 資料切片與報酬區間

### 2.1 1-based 數學記號

為了讓公式乾淨，本文用 1-based 記號：

* 回看區間：$t \in {1,\dots,L}$，其中 $L=60$
* 建倉日：$\tau = 61$
* 平倉日：$\xi = 80$

因此第 $i$ 支股票的 forward return 定義為：

$$
R_i = \frac{P_{i,\xi}}{P_{i,\tau}} - 1
$$

也就是：

$$
R_i = \frac{P_{i,80}}{P_{i,61}} - 1
$$

### 2.2 與原始 CSV 標籤的對照

原始資料的時間標籤是 **0-based**，即：

$$
t_0, t_1, \dots, t_{80}
$$

所以：

* 數學記號中的第 61 天，對應資料列 $t_{60}$
* 數學記號中的第 80 天，對應資料列 $t_{79}$

這只是 **1-based 公式記號** 與 **0-based 資料索引** 的差異，不代表報酬定義錯誤。

---

## 3. 輸入特徵

### 3.1 個股特徵

對第 $i$ 支股票在第 $t$ 天，原始個股特徵記為：

$$
x^{\mathrm{raw}}_{i,t} \in \mathbb{R}^{4}
$$

其中 4 維可寫成：

$$
x^{\mathrm{raw}}_{i,t}
======================

\big[
\text{characteristic}_1,,
\text{characteristic}_2,,
\text{characteristic}*3,,
\text{price}
\big]*{i,t}
$$

### 3.2 市場 FF3 因子

市場因子序列記為：

$$
f^{\mathrm{raw}}_t \in \mathbb{R}^{3}
$$

其中：

$$
f^{\mathrm{raw}}_t = [MKT_t,\ SMB_t,\ HML_t]
$$

---

## 4. 標準化：模型真正吃進去的不是 raw feature

這個專案中，模型不是直接吃 $x^{\mathrm{raw}}_{i,t}$ 與 $f^{\mathrm{raw}}_t$，而是先做標準化。

### 4.1 個股特徵標準化

先用回看窗口前 60 天的統計量做標準化，得到：

$$
x_{i,t} = \mathrm{Standardize}(x^{\mathrm{raw}}_{i,t})
$$

其中標準化所需的均值與標準差，是由該 sample 的 lookback 區間估計得到。

因此，模型實際使用的是：

$$
x_{i,t} \in \mathbb{R}^{4}
$$

而不是 raw feature。

### 4.2 市場因子標準化

同理，市場 FF3 因子也先標準化：

$$
f_t = \mathrm{Standardize}(f^{\mathrm{raw}}_t)
$$

所以模型真正使用的是：

$$
f_t \in \mathbb{R}^{3}
$$

---

## 5. 個股分支的輸入嵌入

先把每支股票每天的標準化後特徵投影到 hidden space：

$$
e_{i,t}^{(0)} = W_x x_{i,t} + b_x
$$

其中：

* $W_x \in \mathbb{R}^{d_s \times 4}$
* $b_x \in \mathbb{R}^{d_s}$

因此：

$$
e_{i,t}^{(0)} \in \mathbb{R}^{d_s}
$$

這裡 $d_s$ 是 stock branch 的 hidden dimension。

---

## 6. 個股分支的時間位置編碼

為了讓 temporal encoder 知道序列中的相對位置，對每個時間點加入時間位置編碼：

$$
p_t^{\mathrm{time}} \in \mathbb{R}^{d_s}
$$

加入後得到：

$$
h_{i,t}^{(0)} = e_{i,t}^{(0)} + p_t^{\mathrm{time}}
$$

因此第 $i$ 支股票在整段回看區間的輸入表示為：

$$
H_i^{(0)}
=========

\begin{bmatrix}
h_{i,1}^{(0)} \
h_{i,2}^{(0)} \
\vdots \
h_{i,L}^{(0)}
\end{bmatrix}
\in \mathbb{R}^{L \times d_s}
$$

其中 $L=60$。

---

## 7. 第一層注意力：Stock Temporal Attention

這一層只沿著**時間維度**做 self-attention，意思是每支股票先看自己的歷史。

對第 $i$ 支股票：

$$
Q_i = H_i^{(0)} W_Q^{(s)}, \qquad
K_i = H_i^{(0)} W_K^{(s)}, \qquad
V_i = H_i^{(0)} W_V^{(s)}
$$

其中：

$$
Q_i, K_i, V_i \in \mathbb{R}^{L \times d_s}
$$

單頭 attention 可寫成：

$$
\mathrm{Attn}_{\mathrm{time}}(H_i^{(0)})
========================================

\mathrm{softmax}!\left(\frac{Q_i K_i^\top}{\sqrt{d_s}}\right)V_i
$$

若使用 multi-head attention，簡潔記為：

$$
H_i = \mathrm{TemporalEncoder}_{\mathrm{stock}}(H_i^{(0)})
$$

其中：

$$
H_i \in \mathbb{R}^{L \times d_s}
$$

這個 $H_i$ 表示第 $i$ 支股票在前 $L=60$ 天上的時序隱表示。

---

## 8. 將每支股票的時間序列壓成摘要向量

temporal encoder 輸出後，將每支股票的整段歷史壓成一個 summary vector。

若使用 mean pooling：

$$
z_i = \frac{1}{L}\sum_{t=1}^{L} H_{i,t}
$$

若使用 last-token pooling：

$$
z_i = H_{i,L}
$$

因此：

$$
z_i \in \mathbb{R}^{d_s}
$$

這個 $z_i$ 是第 $i$ 支股票在 lookback 期間的摘要表示。

---

## 9. 市場 FF3 分支的輸入嵌入

市場 FF3 因子不直接混進個股特徵，而是走獨立分支。

先把標準化後的市場因子投影到 market hidden space：

$$
g_t^{(0)} = W_f f_t + b_f
$$

其中：

* $W_f \in \mathbb{R}^{d_m \times 3}$
* $b_f \in \mathbb{R}^{d_m}$

因此：

$$
g_t^{(0)} \in \mathbb{R}^{d_m}
$$

這裡 $d_m$ 是 market branch 的 hidden dimension。

---

## 10. 市場分支的時間位置編碼

對市場因子序列同樣加入時間位置資訊：

$$
p_t^{\mathrm{market}} \in \mathbb{R}^{d_m}
$$

加入後：

$$
u_t^{(0)} = g_t^{(0)} + p_t^{\mathrm{market}}
$$

整段市場輸入可寫成：

$$
U^{(0)}
=======

\begin{bmatrix}
u_1^{(0)} \
u_2^{(0)} \
\vdots \
u_L^{(0)}
\end{bmatrix}
\in \mathbb{R}^{L \times d_m}
$$

---

## 11. 市場 FF3 的 Temporal Encoder

市場分支也沿時間維度做 self-attention：

$$
Q^{(m)} = U^{(0)} W_Q^{(m)}, \qquad
K^{(m)} = U^{(0)} W_K^{(m)}, \qquad
V^{(m)} = U^{(0)} W_V^{(m)}
$$

$$
\mathrm{Attn}_{\mathrm{market}}(U^{(0)})
========================================

\mathrm{softmax}!\left(\frac{Q^{(m)}(K^{(m)})^\top}{\sqrt{d_m}}\right)V^{(m)}
$$

簡潔記為：

$$
U = \mathrm{TemporalEncoder}_{\mathrm{market}}(U^{(0)})
$$

其中：

$$
U \in \mathbb{R}^{L \times d_m}
$$

再經 pooling 得到市場摘要：

$$
m = \mathrm{Pool}(U)
$$

例如 mean pooling：

$$
m = \frac{1}{L}\sum_{t=1}^{L} U_t
$$

因此：

$$
m \in \mathbb{R}^{d_m}
$$

這個 $m$ 表示市場在前 60 天的摘要狀態。

---

## 12. 融合個股摘要與市場摘要

對每支股票，把自己的摘要 $z_i$ 與市場摘要 $m$ 串接：

$$
z_i' = [z_i ; m]
$$

因此：

$$
z_i' \in \mathbb{R}^{d_s + d_m}
$$

接著投影到橫截面注意力之前的共同表示空間：

$$
u_i = W_c z_i' + b_c
$$

其中：

* $W_c \in \mathbb{R}^{d_c \times (d_s + d_m)}$
* $b_c \in \mathbb{R}^{d_c}$

所以：

$$
u_i \in \mathbb{R}^{d_c}
$$

---

## 13. Stock ID Embedding：專案實作是 concat，不是相加

這裡是本專案與你舊版文檔最重要的差異之一。

每支股票有自己的 identity embedding。建立一個可學習的 embedding table：

$$
E^{\mathrm{id}} \in \mathbb{R}^{N \times d_{\mathrm{id}}}
$$

第 $i$ 支股票的 ID embedding 為：

$$
e_i^{\mathrm{id}} = E^{\mathrm{id}}_{\mathrm{id}(i)}
$$

其中：

$$
e_i^{\mathrm{id}} \in \mathbb{R}^{d_{\mathrm{id}}}
$$

### 正確做法：直接 concat

本專案不是把 $e_i^{\mathrm{id}}$ 加到 $u_i$ 上，而是直接串接：

$$
\tilde{u}_i = [u_i ; e_i^{\mathrm{id}}]
$$

因此：

$$
\tilde{u}*i \in \mathbb{R}^{d_c + d*{\mathrm{id}}}
$$

這表示進入後續 cross-sectional attention 的 token 維度是：

$$
d_{\mathrm{attn}} = d_c + d_{\mathrm{id}}
$$

而不是 $d_c$。

直觀上，這代表模型保留了兩種不同來源的資訊：

* $u_i$：由歷史個股訊號與市場摘要得到的內容表示
* $e_i^{\mathrm{id}}$：股票本身的身份資訊

因為使用 concat，模型可以在後續 attention 中自行學習如何利用這兩部分，而不是事先把它們壓成同一空間後相加。

---

## 14. 第二層注意力：Cross-Sectional Stock Attention

把所有股票的表示堆疊成矩陣：

$$
\tilde{U}
=========

\begin{bmatrix}
\tilde{u}_1 \
\tilde{u}_2 \
\vdots \
\tilde{u}*N
\end{bmatrix}
\in \mathbb{R}^{N \times (d_c + d*{\mathrm{id}})}
$$

然後沿著**股票維度**做 self-attention：

$$
Q^{(c)} = \tilde{U} W_Q^{(c)}, \qquad
K^{(c)} = \tilde{U} W_K^{(c)}, \qquad
V^{(c)} = \tilde{U} W_V^{(c)}
$$

若 attention hidden dimension 仍記作 $d_{\mathrm{attn}} = d_c + d_{\mathrm{id}}$，則：

$$
Q^{(c)}, K^{(c)}, V^{(c)} \in \mathbb{R}^{N \times d_{\mathrm{attn}}}
$$

attention 寫成：

$$
\mathrm{Attn}_{\mathrm{stock}}(\tilde{U})
=========================================

\mathrm{softmax}!\left(\frac{Q^{(c)}(K^{(c)})^\top}{\sqrt{d_{\mathrm{attn}}}}\right)V^{(c)}
$$

簡潔記為：

$$
\hat{U} = \mathrm{CrossSectionalEncoder}(\tilde{U})
$$

其中：

$$
\hat{U}
=======

\begin{bmatrix}
\hat{u}_1 \
\hat{u}_2 \
\vdots \
\hat{u}*N
\end{bmatrix}
\in \mathbb{R}^{N \times d*{\mathrm{attn}}}
$$

這裡的 $\hat{u}_i$ 已融合：

* 股票自己的 60 天歷史
* 市場 FF3 的摘要狀態
* 股票身份資訊
* 股票之間的橫截面互動

---

## 15. 股票打分（Stock Logits）

對每支股票的最終表示做打分：

$$
s_i = w_s^\top \hat{u}_i + b_s
$$

其中：

* $w_s \in \mathbb{R}^{d_{\mathrm{attn}}}$
* $b_s \in \mathbb{R}$

因此所有股票的 logits 為：

$$
s = [s_1,\dots,s_N] \in \mathbb{R}^{N}
$$

---

## 16. 現金部位的打分（Cash Logit）

現金不作為一個 stock token 進入 cross-sectional attention，而是另外由股票整體表示生成。

先對所有股票做 pooling：

$$
\bar{u} = \frac{1}{N}\sum_{i=1}^{N}\hat{u}_i
$$

再由 pooled representation 產生 cash logit：

$$
s_{\mathrm{cash}} = w_{\mathrm{cash}}^\top \bar{u} + b_{\mathrm{cash}}
$$

其中：

* $w_{\mathrm{cash}} \in \mathbb{R}^{d_{\mathrm{attn}}}$
* $b_{\mathrm{cash}} \in \mathbb{R}$

所以：

$$
s_{\mathrm{cash}} \in \mathbb{R}
$$

---

## 17. 最終投組配置

把所有股票 logits 與 cash logit 串接：

$$
\tilde{s} = [s_1,\dots,s_N,s_{\mathrm{cash}}]
$$

再做 softmax：

$$
[w_1,\dots,w_N,w_{\mathrm{cash}}]
=================================

\mathrm{softmax}(\tilde{s})
$$

因此滿足：

$$
\sum_{i=1}^{N} w_i + w_{\mathrm{cash}} = 1
$$

且：

$$
w_i \ge 0, \qquad w_{\mathrm{cash}} \ge 0
$$

這代表模型輸出的是 **long-only 且允許持有現金** 的配置權重。

---

## 18. 持有期報酬

模型在看完前 $L=60$ 天資料後，於 $\tau=61$ 決定配置，持有到 $\xi=80$。

第 $i$ 支股票的持有期報酬為：

$$
R_i = \frac{P_{i,\xi}}{P_{i,\tau}} - 1
$$

即：

$$
R_i = \frac{P_{i,80}}{P_{i,61}} - 1
$$

現金的報酬 baseline 設為：

$$
R_{\mathrm{cash}} = 0
$$

因此投組報酬為：

$$
R_p = \sum_{i=1}^{N} w_i R_i + w_{\mathrm{cash}}R_{\mathrm{cash}}
$$

因為 $R_{\mathrm{cash}}=0$，可簡化為：

$$
R_p = \sum_{i=1}^{N} w_i R_i
$$

---

## 19. 損失函數：理論形式與目前專案實際訓練情境

這一節要區分兩件事：

* 一般數學上可寫的 loss 形式
* 目前專案實際執行時的訓練情境

### 19.1 Return-based objective

若只考慮單一 sample，最直接的目標是最大化該 sample 的投組報酬：

$$
\mathcal{L}_{\mathrm{return}} = -R_p
$$

若有一般性的 batch 記號，才會寫成：

$$
\mathcal{L}*{\mathrm{return}} = -\frac{1}{B}\sum*{b=1}^{B} R_p^{(b)}
$$

但對目前專案來說，核心情境其實是**單一 sample 的更新**，因此更貼近實作的寫法是：

$$
\mathcal{L}_{\mathrm{return}} = -R_p
$$

### 19.2 Sharpe-style objective

理論上，若 batch 中有多個 sample，可定義：

$$
\mu_R = \frac{1}{B}\sum_{b=1}^{B} R_p^{(b)}
$$

$$
\sigma_R =
\sqrt{
\frac{1}{B}\sum_{b=1}^{B}\left(R_p^{(b)}-\mu_R\right)^2
+\varepsilon
}
$$

則 Sharpe-style loss 可寫成：

$$
\mathcal{L}_{\mathrm{sharpe}} = -\frac{\mu_R}{\sigma_R}
$$

### 19.3 目前專案的實際情況

本專案目前不是一般意義下、多個獨立 windows 組成的大 batch 訓練。

實際上：

* 只有一個 cross-boundary analysis window
* `legal_train_windows = 0`
* `legal_test_windows = 0`
* 訓練更接近**單一 sample 的 diagnostic update**

因此目前若選用 sharpe 模式，程式在 batch 太小時，會退回到等價於 $-\text{mean return}$ 的處理；也就是說，現階段並不會真正實現一般 batch-based 的 Sharpe objective。

所以，如果要描述**目前專案實際有效的訓練目標**，更準確的說法是：

$$
\mathcal{L}_{\mathrm{effective}} \approx -R_p
$$

或在實作語意上寫成：

$$
\mathcal{L}_{\mathrm{effective}} \approx -\mathrm{mean}(R_p)
$$

但由於目前只有單一 sample，這兩者在實際效果上是一致的。

---

## 20. 整體流程總公式

### 20.1 個股分支

$$
x^{\mathrm{raw}}*{i,t}
\xrightarrow{\mathrm{Standardize}}
x*{i,t}
\xrightarrow{\mathrm{Linear}}
e_{i,t}^{(0)}
\xrightarrow{+;p_t^{\mathrm{time}}}
h_{i,t}^{(0)}
\xrightarrow{\mathrm{Stock\ Temporal\ Encoder}}
H_i
\xrightarrow{\mathrm{Pool}}
z_i
$$

### 20.2 市場分支

$$
f^{\mathrm{raw}}_t
\xrightarrow{\mathrm{Standardize}}
f_t
\xrightarrow{\mathrm{Linear}}
g_t^{(0)}
\xrightarrow{+;p_t^{\mathrm{market}}}
u_t^{(0)}
\xrightarrow{\mathrm{Market\ Temporal\ Encoder}}
U
\xrightarrow{\mathrm{Pool}}
m
$$

### 20.3 融合與 ID concat

$$
z_i' = [z_i ; m]
$$

$$
u_i = W_c z_i' + b_c
$$

$$
\tilde{u}_i = [u_i ; e_i^{\mathrm{id}}]
$$

### 20.4 股票橫截面 attention

$$
\hat{U} = \mathrm{CrossSectionalEncoder}(\tilde{U})
$$

### 20.5 打分與配置

$$
s_i = w_s^\top \hat{u}_i + b_s
$$

$$
\bar{u} = \frac{1}{N}\sum_{i=1}^{N}\hat{u}_i
$$

$$
s_{\mathrm{cash}} = w_{\mathrm{cash}}^\top \bar{u} + b_{\mathrm{cash}}
$$

$$
[w_1,\dots,w_N,w_{\mathrm{cash}}]
=================================

\mathrm{softmax}([s_1,\dots,s_N,s_{\mathrm{cash}}])
$$

### 20.6 報酬與 loss

$$
R_i = \frac{P_{i,80}}{P_{i,61}} - 1
$$

$$
R_p = \sum_{i=1}^{N} w_i R_i
$$

目前專案的有效訓練目標可近似寫成：

$$
\mathcal{L}_{\mathrm{effective}} \approx -R_p
$$

---

## 21. 直觀解讀

這個模型可以口語化理解成以下幾步：

1. **個股 temporal encoder**
   每支股票先看自己前 60 天的歷史，整理成個股摘要 $z_i$。

2. **市場 temporal encoder**
   全市場 FF3 在前 60 天的變化，被整理成市場摘要 $m$。

3. **融合個股與市場資訊**
   每支股票的決策不只依賴自己，也依賴同一段期間的市場狀態。

4. **股票身份資訊不是相加，而是 concat**
   模型不只是知道「最近 60 天長什麼樣」，也明確保留「這是哪一支股票」。

5. **cross-sectional attention**
   股票彼此互相看，學習橫截面上的相對關係。

6. **cash head**
   模型不只決定各股票權重，也決定是否保留一部分權重在現金。

7. **以未來持有期報酬訓練**
   用第 61 天建倉到第 80 天平倉的 forward return 來回頭訓練配置模型。

8. **目前訓練是單一 window 的 diagnostic update**
   因此實際有效 loss 更接近單一 sample return objective，而不是穩定的大 batch Sharpe optimization。

---

## 22. 建議你在 README 再額外加一段「常見誤解」

你可以直接附下面這段：

### 常見誤解 1：$T=60$ 與 sample 長度 $81$ 是同一件事嗎？

不是。

* $L=60$ 是模型的 lookback 長度
* $S=81$ 是整段 sample 的總時間長度

### 常見誤解 2：Stock ID embedding 是加法還是串接？

本專案是 **串接（concat）**，不是加法。

### 常見誤解 3：模型吃的是 raw feature 嗎？

不是。模型吃的是**先用 lookback 區間做標準化後的特徵**。

### 常見誤解 4：Sharpe loss 真的在目前訓練中被完整實現嗎？

目前不是典型的大 batch Sharpe optimization。由於實際上只有單一 analysis window，程式在 batch 太小時會退回到接近 return-based 的更新。


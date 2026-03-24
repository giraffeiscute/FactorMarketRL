## 1. 整體流程總公式

對單一 sample，觀察 $N$ 支股票。其時間參數由該資料分段（Split）的總天數 $T$ 與分析區間 $H$（對應配置參數 analysis_horizon_days）動態決定：

$$
L = T - H,\qquad \tau = L + 1,\qquad \xi = T
$$

其中 $L$ 為回看長度（lookback），$\tau$ 為建倉時點（entry），$\xi$ 為平倉時點（exit）。

原始輸入張量：

$$
X^{\mathrm{raw}} \in \mathbb{R}^{N \times L \times 4},\qquad
F^{\mathrm{raw}} \in \mathbb{R}^{L \times 3}
$$

標準化後：

$$
X=\mathrm{Standardize}(X^{\mathrm{raw}})\in\mathbb{R}^{N\times L\times 4}
$$

$$
F=\mathrm{Standardize}(F^{\mathrm{raw}})\in\mathbb{R}^{L\times 3}
$$
(X是個股資訊，F是市場資訊)

---

## 1.1 個股時間分支

對第 $i$ 支股票、第 $t$ 個時間點：

$$
x_{i,t}\in\mathbb{R}^{4}
$$

線性投影：

$$
e_{i,t}^{(0)} = W_x x_{i,t} + b_x
$$

$$
W_x\in\mathbb{R}^{d_s\times 4},\qquad
b_x\in\mathbb{R}^{d_s},\qquad
e_{i,t}^{(0)}\in\mathbb{R}^{d_s}
$$

加入時間位置編碼：

$$
h_{i,t}^{(0)} = e_{i,t}^{(0)} + p_t^{\mathrm{time}}
$$

$$
p_t^{\mathrm{time}}\in\mathbb{R}^{d_s},\qquad
h_{i,t}^{(0)}\in\mathbb{R}^{d_s}
$$

第 $i$ 支股票的時間序列表示：

$$
H_i^{(0)}=
\begin{bmatrix}
h_{i,1}^{(0)}\
h_{i,2}^{(0)}\
\vdots\
h_{i,L}^{(0)}
\end{bmatrix}
\in\mathbb{R}^{L\times d_s}
$$

temporal self-attention：

$$
Q_i = H_i^{(0)}W_Q^{(s)},\qquad
K_i = H_i^{(0)}W_K^{(s)},\qquad
V_i = H_i^{(0)}W_V^{(s)}
$$

$$
Q_i,K_i,V_i\in\mathbb{R}^{L\times d_s}
$$

$$
H_i=\mathrm{softmax}\left(\frac{Q_iK_i^\top}{\sqrt{d_s}}\right)V_i
$$

$$
H_i\in\mathbb{R}^{L\times d_s}
$$

時間池化後得到個股摘要向量：

$$
z_i=\mathrm{Pool}(H_i)\in\mathbb{R}^{d_s}
$$

全部股票堆疊後：

$$
Z=
\begin{bmatrix}
z_1\
z_2\
\vdots\
z_N
\end{bmatrix}
\in\mathbb{R}^{N\times d_s}
$$

---

## 1.2 市場時間分支

對第 $t$ 個時間點：

$$
f_t\in\mathbb{R}^{3}
$$

線性投影：

$$
g_t^{(0)} = W_f f_t + b_f
$$

$$
W_f\in\mathbb{R}^{d_m\times 3},\qquad
b_f\in\mathbb{R}^{d_m},\qquad
g_t^{(0)}\in\mathbb{R}^{d_m}
$$

加入市場位置編碼：

$$
u_t^{(0)} = g_t^{(0)} + p_t^{\mathrm{market}}
$$

$$
p_t^{\mathrm{market}}\in\mathbb{R}^{d_m},\qquad
u_t^{(0)}\in\mathbb{R}^{d_m}
$$

市場序列表示：

$$
U^{(0)}=
\begin{bmatrix}
u_1^{(0)}\
u_2^{(0)}\
\vdots\
u_L^{(0)}
\end{bmatrix}
\in\mathbb{R}^{L\times d_m}
$$

market temporal self-attention：

$$
Q^{(m)} = U^{(0)}W_Q^{(m)},\qquad
K^{(m)} = U^{(0)}W_K^{(m)},\qquad
V^{(m)} = U^{(0)}W_V^{(m)}
$$

$$
Q^{(m)},K^{(m)},V^{(m)}\in\mathbb{R}^{L\times d_m}
$$

$$
U=\mathrm{softmax}\left(\frac{Q^{(m)}(K^{(m)})^\top}{\sqrt{d_m}}\right)V^{(m)}
$$

$$
U\in\mathbb{R}^{L\times d_m}
$$

池化後得到市場摘要向量：

$$
m=\mathrm{Pool}(U)\in\mathbb{R}^{d_m}
$$


---

## 1.3 個股摘要與市場摘要融合

對第 $i$ 支股票：

$$
z_i'=[z_i;m]
$$

$$
z_i'\in\mathbb{R}^{d_s+d_m}
$$

線性融合：

$$
u_i=W_c z_i' + b_c
$$

$$
W_c\in\mathbb{R}^{d_c\times(d_s+d_m)},\qquad
b_c\in\mathbb{R}^{d_c},\qquad
u_i\in\mathbb{R}^{d_c}
$$

全部股票堆疊後：

$$
U_c=
\begin{bmatrix}
u_1\
u_2\
\vdots\
u_N
\end{bmatrix}
\in\mathbb{R}^{N\times d_c}
$$

---

## 1.4 Stock ID Embedding 與 concat

stock ID embedding table：

$$
E^{\mathrm{id}}\in\mathbb{R}^{N\times d_{\mathrm{id}}}
$$

第 $i$ 支股票的 ID embedding：

$$
e_i^{\mathrm{id}} = E^{\mathrm{id}}_{\mathrm{id}(i)}
$$

$$
e_i^{\mathrm{id}}\in\mathbb{R}^{d_{\mathrm{id}}}
$$

與內容表示 concat：

$$
\tilde{u}_i=[u_i;e_i^{\mathrm{id}}]
$$

$$
\tilde{u}*i\in\mathbb{R}^{d_c+d*{\mathrm{id}}}
$$

定義：

$$
d_{\mathrm{attn}}=d_c+d_{\mathrm{id}}
$$

全部股票堆疊後：

$$
\tilde{U}=
\begin{bmatrix}
\tilde{u}_1\
\tilde{u}_2\
\vdots\
\tilde{u}*N
\end{bmatrix}
\in\mathbb{R}^{N\times d*{\mathrm{attn}}}
$$

---

## 1.5 Cross-Sectional Stock Attention

橫截面 self-attention：

$$
Q^{(c)}=\tilde{U}W_Q^{(c)},\qquad
K^{(c)}=\tilde{U}W_K^{(c)},\qquad
V^{(c)}=\tilde{U}W_V^{(c)}
$$

$$
Q^{(c)},K^{(c)},V^{(c)}\in\mathbb{R}^{N\times d_{\mathrm{attn}}}
$$

$$
\hat{U}=\mathrm{softmax}\left(\frac{Q^{(c)}(K^{(c)})^\top}{\sqrt{d_{\mathrm{attn}}}}\right)V^{(c)}
$$

$$
\hat{U}=
\begin{bmatrix}
\hat{u}_1\
\hat{u}_2\
\vdots\
\hat{u}*N
\end{bmatrix}
\in\mathbb{R}^{N\times d*{\mathrm{attn}}}
$$

---

## 1.6 股票分數與現金分數

對第 $i$ 支股票：

$$
s_i=w_s^\top \hat{u}_i+b_s
$$

$$
w_s\in\mathbb{R}^{d_{\mathrm{attn}}},\qquad
b_s\in\mathbb{R},\qquad
s_i\in\mathbb{R}
$$

全部股票分數向量：

$$
s=[s_1,\dots,s_N]\in\mathbb{R}^{N}
$$

股票池化表示：

$$
\bar{u}=\frac{1}{N}\sum_{i=1}^{N}\hat{u}_i
$$

$$
\bar{u}\in\mathbb{R}^{d_{\mathrm{attn}}}
$$

現金分數：

$$
s_{\mathrm{cash}}=w_{\mathrm{cash}}^\top\bar{u}+b_{\mathrm{cash}}
$$

$$
w_{\mathrm{cash}}\in\mathbb{R}^{d_{\mathrm{attn}}},\qquad
b_{\mathrm{cash}}\in\mathbb{R},\qquad
s_{\mathrm{cash}}\in\mathbb{R}
$$

---

## 1.7 最終配置權重

將股票分數與現金分數串接：

$$
\tilde{s}=[s_1,\dots,s_N,s_{\mathrm{cash}}]\in\mathbb{R}^{N+1}
$$

softmax 後得到最終配置：

$$
[w_1,\dots,w_N,w_{\mathrm{cash}}]=\mathrm{softmax}(\tilde{s})
$$

$$
w\in\mathbb{R}^{N+1}
$$

$$
\sum_{i=1}^{N}w_i+w_{\mathrm{cash}}=1
$$

$$
w_i\ge 0,\qquad w_{\mathrm{cash}}\ge 0
$$

---

## 1.8 持有期報酬

第 $i$ 支股票的 forward return：

$$
R_i=\frac{P_{i,\xi}}{P_{i,\tau}}-1
$$

$$
R_i=\frac{P_{i,80}}{P_{i,61}}-1
$$

$$
R_i\in\mathbb{R}
$$

全部股票報酬向量：

$$
R=[R_1,\dots,R_N]\in\mathbb{R}^{N}
$$

現金報酬：

$$
R_{\mathrm{cash}}=0
$$

投組報酬：

$$
R_p=\sum_{i=1}^{N}w_iR_i+w_{\mathrm{cash}}R_{\mathrm{cash}}
$$

$$
R_p=\sum_{i=1}^{N}w_iR_i
$$

$$
R_p\in\mathbb{R}
$$

---

## 1.9 損失函數

單一 sample：

$$
\mathcal{L}_{\mathrm{return}}=-R_p
$$

batch 形式：

$$
\mathcal{L}*{\mathrm{return}}=-\frac{1}{B}\sum*{b=1}^{B}R_p^{(b)}
$$

Sharpe-style 形式：

$$
\mu_R=\frac{1}{B}\sum_{b=1}^{B}R_p^{(b)}
$$

$$
\sigma_R=
\sqrt{
\frac{1}{B}\sum_{b=1}^{B}\left(R_p^{(b)}-\mu_R\right)^2+\varepsilon
}
$$

$$
\mathcal{L}_{\mathrm{sharpe}}=-\frac{\mu_R}{\sigma_R}
$$

目前有效訓練目標：

$$
\mathcal{L}_{\mathrm{effective}}\approx -R_p
$$

---

## 1.10 總流程串接公式

逐股票路徑：

$$
x^{\mathrm{raw}}*{i,t}
\rightarrow
x*{i,t}
\rightarrow
e_{i,t}^{(0)}
\rightarrow
h_{i,t}^{(0)}
\rightarrow
H_i
\rightarrow
z_i
\rightarrow
[z_i;m]
\rightarrow
u_i
\rightarrow
[u_i;e_i^{\mathrm{id}}]
\rightarrow
\tilde{u}_i
\rightarrow
\hat{u}_i
\rightarrow
s_i
$$

市場路徑：

$$
f_t^{\mathrm{raw}}
\rightarrow
f_t
\rightarrow
g_t^{(0)}
\rightarrow
u_t^{(0)}
\rightarrow
U
\rightarrow
m
$$

最終配置與訓練路徑：

$$
[s_1,\dots,s_N,s_{\mathrm{cash}}]
\rightarrow
[w_1,\dots,w_N,w_{\mathrm{cash}}]
\rightarrow
R_p
\rightarrow
\mathcal{L}_{\mathrm{effective}}
$$

---

## 1.11 矩陣總覽版

$$
X^{\mathrm{raw}}\in\mathbb{R}^{N\times L\times 4}
;\xrightarrow{\mathrm{Standardize}};
X\in\mathbb{R}^{N\times L\times 4}
;\xrightarrow{\mathrm{Linear}};
E^{(0)}\in\mathbb{R}^{N\times L\times d_s}
$$

$$
E^{(0)}
;\xrightarrow{+;P^{\mathrm{time}}};
H^{(0)}\in\mathbb{R}^{N\times L\times d_s}
;\xrightarrow{\mathrm{Temporal\ Encoder}};
H\in\mathbb{R}^{N\times L\times d_s}
;\xrightarrow{\mathrm{Pool}};
Z\in\mathbb{R}^{N\times d_s}
$$

$$
F^{\mathrm{raw}}\in\mathbb{R}^{L\times 3}
;\xrightarrow{\mathrm{Standardize}};
F\in\mathbb{R}^{L\times 3}
;\xrightarrow{\mathrm{Linear}};
G^{(0)}\in\mathbb{R}^{L\times d_m}
;\xrightarrow{+;P^{\mathrm{market}}};
U^{(0)}\in\mathbb{R}^{L\times d_m}
$$

$$
U^{(0)}
;\xrightarrow{\mathrm{Market\ Temporal\ Encoder}};
U\in\mathbb{R}^{L\times d_m}
;\xrightarrow{\mathrm{Pool}};
m\in\mathbb{R}^{d_m}
$$

$$
Z'=[Z;\mathbf{1}_N m^\top]\in\mathbb{R}^{N\times(d_s+d_m)}
;\xrightarrow{\mathrm{Linear}};
U_c\in\mathbb{R}^{N\times d_c}
$$

$$
E^{\mathrm{id}}\in\mathbb{R}^{N\times d_{\mathrm{id}}},
\qquad
\tilde{U}=[U_c;E^{\mathrm{id}}]\in\mathbb{R}^{N\times d_{\mathrm{attn}}}
$$

$$
\tilde{U}
;\xrightarrow{\mathrm{CrossSectional\ Encoder}};
\hat{U}\in\mathbb{R}^{N\times d_{\mathrm{attn}}}
;\xrightarrow{\mathrm{Score}};
s\in\mathbb{R}^{N}
$$

$$
\bar{u}=\frac{1}{N}\sum_{i=1}^{N}\hat{u}*i\in\mathbb{R}^{d*{\mathrm{attn}}},
\qquad
s_{\mathrm{cash}}\in\mathbb{R}
$$

$$
w=\mathrm{softmax}([s;s_{\mathrm{cash}}])\in\mathbb{R}^{N+1}
$$

$$
R_i=\frac{P_{i,80}}{P_{i,61}}-1,\qquad
R_p=\sum_{i=1}^{N}w_iR_i,\qquad
\mathcal{L}_{\mathrm{effective}}\approx -R_p
$$


# FactorMarketRL README

## 1. 專案簡介

這個專案是一個以 PyTorch 撰寫的投資組合最佳化 toy model。它的目的不是回測真實市場資料，也不是建立可直接上線的交易系統，而是用一套可控的模擬環境，觀察在不同風險報酬目標下，模型會學出什麼樣的股票權重配置。

依照目前程式碼的實作，這個專案**沒有使用真實市場資料**。所有股票特徵、股票年化報酬、波動率、每日報酬序列，都是由程式在執行時人工生成：

- `stock_embeddings` 是隨機產生的股票 embedding。
- `annual_returns` 是人工設定的報酬結構。
- `annual_volatility` 是人工指定的三段波動率。
- `daily_returns` 是根據上述平均數與波動率，用常態分布隨機抽樣出來的模擬日報酬。

因此，這個專案的核心任務可以描述為：

> 在一個人工設計的模擬市場中，訓練一個簡單的神經網路，根據每檔股票的 embedding 輸出投資權重，並在不同 loss function 下比較投資組合的風險報酬表現。

就分類來看，這個專案比較接近：

- `toy model`
- `simulation`
- `portfolio optimization`

它**不是**真實市場資料驅動的 reinforcement learning 專案，也**不是**動態時序交易策略。模型學到的是一個**靜態配置規則**：輸入股票 embedding，輸出該股票在投組中的權重。

## 2. 專案結構

目前專案根目錄下與主流程有關的檔案如下：

### 核心程式檔

- `Toy_Model_Loss_Simulation.py`
  - 專案主入口。
  - 定義所有 loss function。
  - 定義 `train_and_evaluate(...)`、`run_experiment(...)`、`main(config)`。
  - 負責訓練、評估、寫報告、畫圖、建立輸出資料夾。

- `market_environment.py`
  - 定義 `MarketEnvironment`。
  - 負責建立模擬市場環境。
  - 生成股票 embedding、年化報酬、年化波動率、日報酬分布。

- `portfolio_model.py`
  - 定義 `PortfolioModel`。
  - 是一個非常簡單的前饋神經網路，將股票 embedding 映射成投資權重。

- `performance_metrics.py`
  - 定義 `PerformanceMetrics`。
  - 接收投組日報酬與股票池日報酬，計算年化報酬、Sharpe、Sortino、最大回撤、CVaR、Beta、IR。

### 輸出與附加資源

- `result/`
  - 所有執行結果的輸出目錄。
  - 每次執行都會在底下建立一個新資料夾，例如 `results_20260310-152952/`。

- `result/results_時間戳/`
  - 單次實驗的輸出目錄。
  - 通常包含：
    - `performance_report.txt`
    - 多張 `weights_*.png`
    - 一張 `cumulative_returns_*.png`

- `SourceHanSansSC-Regular.otf`
  - 用來支援 Matplotlib 畫圖時顯示中文。
  - 只影響圖表字型，不影響核心計算流程。

## 3. 整體執行流程

這個專案的執行流程是單一路徑，由上到下很清楚。入口是 `Toy_Model_Loss_Simulation.py`，程式執行後會走到 `if __name__ == '__main__':`，讀取檔案底部定義的 `config`，再呼叫 `main(config)`。

### 第一步：進入 `main(config)`

`main(config)` 會先做三件事：

1. 找到目前專案所在目錄。
2. 建立 `result/` 資料夾。
3. 依照目前時間建立一個新的輸出資料夾，例如 `result/results_20260310-152952/`。

之後它會偵測目前是否有 GPU 可用，決定使用 `cuda` 還是 `cpu`，再把 `config` 裡的超參數傳給 `run_experiment(...)`。

### 第二步：進入 `run_experiment(...)`

`run_experiment(...)` 是單次實驗的總控函式。它會先建立一個 `MarketEnvironment(trading_days=trading_days_per_epoch)`，這代表整個實驗所使用的模擬市場環境。

建立環境後，它會定義一組 loss functions。這些 loss 彼此不是加總成一個總 loss，而是：

- 每一個 loss 單獨訓練一個模型
- 每個模型各自產生一組投資權重
- 最後再把不同 loss 的結果拿來比較

`run_experiment(...)` 會對 `loss_functions` 字典中的每個項目，逐一呼叫 `train_and_evaluate(...)`。每跑完一個 loss，就會：

- 拿到該 loss 訓練出的完整股票權重
- 拿到該權重在評估資料上的累積報酬
- 拿到績效指標 `performance`
- 把績效寫進 `performance_report.txt`
- 產生對應的權重分布圖 `weights_*.png`

所有 loss 都跑完之後，`run_experiment(...)` 會再把各 loss 的累積報酬畫在同一張圖上，輸出 `cumulative_returns_*.png`。

### 第三步：進入 `train_and_evaluate(...)`

`train_and_evaluate(...)` 是核心訓練與評估流程。它會：

1. 建立 `PortfolioModel(env.d_model)`。
2. 從 `env` 取出所有股票的 embedding。
3. 建立 `Adam` optimizer。
4. 開始 epoch 迴圈。

在每個 epoch 裡，又會進一步跑 `batch_size` 次模擬。每次模擬都會：

1. 呼叫 `env.get_daily_returns(env.trading_days)` 重新生成一份股票日報酬矩陣。
2. 從所有股票中隨機抽取 `sample_size` 檔股票。
3. 取出這些股票的 embedding 和日報酬。
4. 把 `sampled_embeddings` 丟進模型，得到 `portfolio_weights`。
5. 用 `sampled_returns.T @ portfolio_weights` 算出投組每日報酬 `portfolio_daily_returns`。
6. 把投組每日報酬送入指定的 `loss_function`，得到 loss。
7. 對 loss 做 `backward()`，累積梯度。

一個 epoch 的 `batch_size` 次模擬跑完後，才做一次 `optimizer.step()` 更新參數。

### 第四步：訓練完成後進行評估

`train_and_evaluate(...)` 完成訓練後，不會用訓練時抽樣的 `sample_size` 股票來評估，而是：

1. 對**完整股票池**的 embedding 做一次 forward，得到 `full_weights`。
2. 再呼叫 `env.get_daily_returns(env.trading_days * 3)`，生成一份較長期間的評估資料。
3. 用完整股票池的日報酬乘上完整權重，得到 `portfolio_eval_returns`。
4. 把 `portfolio_eval_returns` 和 `eval_returns_stocks` 丟給 `PerformanceMetrics.calculate_metrics(...)` 算績效。
5. 再把 `portfolio_eval_returns` 轉成 `cumulative_returns`。

最後，`train_and_evaluate(...)` 回傳三個東西：

- `full_weights`
- `cumulative_returns`
- `performance`

### 第五步：輸出圖表與文字報告

整個 pipeline 結束後，輸出內容包含：

- 每個 loss 各自的權重圖
- 一份整體績效報告 `performance_report.txt`
- 一張比較所有 loss 的累積收益圖

所有輸出都會放到 `result/results_時間戳/` 底下。

## 4. 資料流 / Data Flow

下面直接用資料流方式整理整個專案的資訊流動：

### 4.1 入口參數流

`config`  
→ `main(config)`  
→ `run_experiment(...)`  
→ `train_and_evaluate(...)`

`config` 內目前實際使用的欄位有：

- `trading_days_per_epoch`
- `learning_rate`
- `max_epochs`
- `patience`
- `min_delta`
- `batch_size`
- `sample_size`

### 4.2 市場資料生成流

`MarketEnvironment(trading_days)`  
→ 生成 `stock_embeddings`  
→ 生成 `annual_returns`  
→ 生成 `annual_volatility`  
→ 推導 `daily_returns_mean`  
→ 推導 `daily_volatility`

之後每次需要模擬資料時：

`get_daily_returns(num_days)`  
→ 依照 `daily_returns_mean` 與 `daily_volatility`  
→ 從常態分布抽樣  
→ 產生 `daily_returns`

### 4.3 訓練資料流

`stock_embeddings`  
→ 依照隨機抽樣索引取出 `sampled_embeddings`

`daily_returns`  
→ 依照相同索引取出 `sampled_returns`

`sampled_embeddings`  
→ `PortfolioModel.forward(...)`  
→ `portfolio_weights`

`sampled_returns + portfolio_weights`  
→ `torch.matmul(sampled_returns.T, portfolio_weights)`  
→ `portfolio_daily_returns`

`portfolio_daily_returns`  
→ `loss_function(...)`  
→ `loss`

`loss`  
→ `backward()`  
→ 梯度  
→ `optimizer.step()`  
→ 更新模型參數

### 4.4 評估資料流

完整 `stock_embeddings`  
→ `PortfolioModel.forward(...)`  
→ `full_weights`

`env.get_daily_returns(env.trading_days * 3)`  
→ `eval_returns_stocks`

`eval_returns_stocks + full_weights`  
→ `torch.matmul(eval_returns_stocks.T, full_weights)`  
→ `portfolio_eval_returns`

`portfolio_eval_returns + eval_returns_stocks`  
→ `PerformanceMetrics.calculate_metrics(...)`  
→ `performance dict`

`portfolio_eval_returns`  
→ `torch.cumprod(1 + portfolio_eval_returns, dim=0)`  
→ `cumulative_returns`

### 4.5 輸出流

`weights`  
→ 權重分布圖 `weights_*.png`

`performance dict`  
→ `performance_report.txt`

`cumulative_returns`  
→ 累積收益圖 `cumulative_returns_*.png`

## 5. 每個 class / function 的輸入、輸出、作用

本節依照目前程式中的主要 class 與 function，逐一整理其輸入、輸出與角色。

### `MarketEnvironment.__init__(trading_days=252)`

**輸入**

- `trading_days: int`
  - 一年使用多少個交易日，預設為 `252`。

**輸出**

- 無直接 return。
- 會建立以下屬性：
  - `num_stocks_per_regime = 2001`
  - `num_stocks = 6003`
  - `d_model = 64`
  - `stock_embeddings`
  - `annual_returns`
  - `annual_volatility`
  - `daily_returns_mean`
  - `daily_volatility`

**作用**

- 建立整個模擬市場環境。
- 是所有後續資料生成的基礎。

### `MarketEnvironment._generate_returns()`

**輸入**

- 無。

**輸出**

- `np.ndarray`
- shape: `(6003,)`

**作用**

- 生成完整股票池的年化報酬向量。
- 由三個 regime 拼接而成，每個 regime 有 2001 檔股票。

### `MarketEnvironment._generate_single_regime_returns()`

**輸入**

- 無。

**輸出**

- `np.ndarray`
- shape: `(2001,)`

**作用**

- 生成單一 regime 的年化報酬結構。
- 中間股票的年化報酬最高為 `15%`，往左右兩端線性下降，最低為 `-5%`。

### `MarketEnvironment._generate_volatilities()`

**輸入**

- 無。

**輸出**

- `np.ndarray`
- shape: `(6003,)`

**作用**

- 為三個 regime 指定固定年化波動率：
  - regime 1: `5%`
  - regime 2: `10%`
  - regime 3: `15%`

### `MarketEnvironment.get_daily_returns(num_days)`

**輸入**

- `num_days: int`

**輸出**

- `torch.FloatTensor`
- shape: `(6003, num_days)`

**作用**

- 使用 `daily_returns_mean` 與 `daily_volatility` 作為常態分布參數，隨機生成每檔股票的日報酬序列。
- 這是訓練和評估時真正使用的市場報酬資料。

### `PortfolioModel.__init__(d_model)`

**輸入**

- `d_model: int`
  - 目前由 `MarketEnvironment` 提供，值為 `64`。

**輸出**

- 無直接 return。
- 建立模型層：
  - `Linear(d_model, 32)`
  - `ReLU()`
  - `Linear(32, 1)`
  - `Softmax(dim=0)`

**作用**

- 建立從股票 embedding 到投組權重的映射模型。

### `PortfolioModel.forward(embeddings)`

**輸入**

- `embeddings`
  - shape: `(N, 64)`
  - `N` 代表本次輸入的股票數量。

**輸出**

- `weights`
  - shape: `(N,)`

**作用**

- 對每一檔股票計算一個分數，再透過 `Softmax(dim=0)` 轉成權重。
- 輸出權重總和為 `1`。
- 由於是 softmax，這個模型代表的是 long-only 權重分配。

### `PerformanceMetrics.__init__(trading_days=252, risk_free_rate=0.0)`

**輸入**

- `trading_days: int`
- `risk_free_rate: float`

**輸出**

- 無直接 return。

**作用**

- 儲存績效計算所需的年化交易日數與無風險利率。
- 其中 `risk_free_rate` 會先轉換成日頻。

### `PerformanceMetrics.calculate_metrics(portfolio_returns, daily_stock_returns)`

**輸入**

- `portfolio_returns`
  - 型別可為 `torch.Tensor` 或 `np.ndarray`
  - shape: `(T,)` 或可展平成 `(T,)`
- `daily_stock_returns`
  - 型別可為 `torch.Tensor` 或 `np.ndarray`
  - shape: `(N, T)`

**輸出**

- `dict`
  - 包含：
    - `Annualized Return`
    - `Sharpe Ratio`
    - `Sortino Ratio`
    - `Max Drawdown`
    - `CVaR (5%)`
    - `Beta`
    - `IR`

**作用**

- 根據投組日報酬計算整組績效指標。
- `Beta` 的 market proxy 是所有股票日報酬的橫截面平均。
- `IR` 目前實作上直接等於 `Sharpe Ratio`，不是獨立計算的資訊比率。

### `loss_return(p)`

**輸入**

- `p: torch.Tensor`
- shape: `(T,)`

**輸出**

- scalar tensor

**作用**

- 回傳 `-mean(p)`。
- 最小化這個 loss 等於最大化平均報酬。

### `loss_sharpe(p)`

**輸入**

- `p: torch.Tensor`
- shape: `(T,)`

**輸出**

- scalar tensor

**作用**

- 回傳 `-(mean / std)`。
- 最小化等於最大化 Sharpe 型目標。

### `loss_sortino(p)`

**輸入**

- `p: torch.Tensor`
- shape: `(T,)`

**輸出**

- scalar tensor

**作用**

- 只對負報酬的波動做懲罰。
- 若沒有負報酬，直接回傳 `-mean_return`。

### `loss_mdd(p)`

**輸入**

- `p: torch.Tensor`
- shape: `(T,)`

**輸出**

- scalar tensor

**作用**

- 先把日報酬轉為累積報酬，再計算回撤序列。
- 回傳最小 drawdown，也就是最大回撤。

### `loss_cvar(p)`

**輸入**

- `p: torch.Tensor`
- shape: `(T,)`

**輸出**

- scalar tensor

**作用**

- 取 5% 分位數以下的報酬平均值。
- 回傳的是尾部最差情況下的平均報酬。

### `loss_return_vol(p, lam=0.5)`

**輸入**

- `p: torch.Tensor`
- shape: `(T,)`
- `lam: float`

**輸出**

- scalar tensor

**作用**

- 目標函數為 `-(mean - lam * std)`。
- 同時鼓勵高報酬與低波動。

### `loss_return_cvar(p, lam=0.5)`

**輸入**

- `p: torch.Tensor`
- shape: `(T,)`
- `lam: float`

**輸出**

- scalar tensor

**作用**

- 目標函數為 `-(mean + lam * loss_cvar(p))`。
- 由於 `loss_cvar(p)` 通常是負值，這一項等於把尾部風險納入懲罰。

### `loss_sharpe_sortino(p, a=0.5)`

**輸入**

- `p: torch.Tensor`
- shape: `(T,)`
- `a: float`

**輸出**

- scalar tensor

**作用**

- 以線性組合的方式同時考慮 Sharpe 與 Sortino。

### `train_and_evaluate(loss_function, loss_name, env, learning_rate, device, sample_size, batch_size, max_epochs, patience, min_delta)`

**輸入**

- `loss_function`
- `loss_name: str`
- `env: MarketEnvironment`
- `learning_rate: float`
- `device: torch.device`
- `sample_size: int`
- `batch_size: int`
- `max_epochs: int`
- `patience: int`
- `min_delta: float`

**輸出**

- `full_weights`
  - `np.ndarray`
  - shape: `(6003,)`
- `cumulative_returns`
  - `np.ndarray`
  - shape: `(3 * trading_days,)`
- `performance`
  - `dict`

**作用**

- 單一 loss 的訓練與評估總流程。
- 先訓練，再用完整股票池做評估。

### `run_experiment(trading_days_per_epoch, learning_rate, max_epochs, patience, min_delta, device, batch_size, sample_size, output_dir)`

**輸入**

- 訓練所需的超參數與輸出資料夾路徑。

**輸出**

- 無直接 return。

**作用**

- 建立市場環境。
- 逐一跑所有 loss function。
- 整理結果、寫報告、輸出圖檔。

### `main(config)`

**輸入**

- `config: dict`

**輸出**

- 無直接 return。

**作用**

- 專案主入口的控制函式。
- 建立輸出資料夾、選擇裝置、呼叫 `run_experiment(...)`。

## 6. 維度 / shape 說明

本專案最重要的張量與陣列 shape 如下。

### 市場環境相關

- `stock_embeddings`
  - shape: `(6003, 64)`
  - 含義：6003 檔股票，每檔股票一個 64 維 embedding。

- `annual_returns`
  - shape: `(6003,)`
  - 含義：每檔股票的年化期望報酬。

- `annual_volatility`
  - shape: `(6003,)`
  - 含義：每檔股票的年化波動率。

- `daily_returns_mean`
  - shape: `(6003,)`
  - 含義：每檔股票的日平均報酬。

- `daily_volatility`
  - shape: `(6003,)`
  - 含義：每檔股票的日波動率。

- `get_daily_returns(num_days)` 的輸出
  - shape: `(6003, num_days)`
  - 含義：6003 檔股票在 `num_days` 天內的模擬日報酬矩陣。

### 訓練階段相關

- `sampled_embeddings`
  - shape: `(sample_size, 64)`
  - 含義：本次訓練抽樣的股票 embedding。

- `sampled_returns`
  - shape: `(sample_size, trading_days)`
  - 含義：本次抽樣股票在訓練期間的日報酬。

- `portfolio_weights`
  - shape: `(sample_size,)`
  - 含義：模型對本次抽樣股票輸出的權重。

- `portfolio_daily_returns`
  - shape: `(trading_days,)`
  - 含義：該組抽樣股票形成的投組每日報酬。

### 評估階段相關

- `full_weights`
  - shape: `(6003,)`
  - 含義：模型在完整股票池上的權重輸出。

- `eval_returns_stocks`
  - shape: `(6003, 3 * trading_days)`
  - 含義：評估期間的股票日報酬。

- `portfolio_eval_returns`
  - shape: `(3 * trading_days,)`
  - 含義：完整股票池權重在評估資料上的投組每日報酬。

- `cumulative_returns`
  - shape: `(3 * trading_days,)`
  - 含義：由投組日報酬累乘得到的累積報酬曲線。

## 7. 訓練邏輯

### 每個 epoch 在做什麼

每個 epoch 代表一次參數更新週期。在一個 epoch 中，程式不只做一次抽樣，而是做 `batch_size` 次獨立模擬。每次模擬都會重新生成一份日報酬資料，再從股票池中隨機抽 `sample_size` 檔股票做訓練。

也就是說，一個 epoch 的流程是：

1. 清空梯度 `optimizer.zero_grad()`
2. 重複 `batch_size` 次：
   - 重新生成模擬日報酬
   - 抽股票
   - 算權重
   - 算投組報酬
   - 算 loss
   - `loss.backward()`
3. 做一次 `optimizer.step()`

### `batch_size` 在這份程式中的意思

這裡的 `batch_size` 不是傳統監督學習中把固定資料集切成 batch 的意思，而是：

> 每個 epoch 要累積幾次獨立的模擬樣本，再做一次參數更新。

換句話說，它比較像是 Monte Carlo 樣本數，而不是資料集切批大小。

### `sample_size` 在這份程式中的意思

`sample_size` 表示每次訓練時，從 6003 檔股票中隨機抽多少檔股票來組成當次訓練的子集合。這些股票的 embedding 會輸入模型，這些股票的日報酬也會被拿來計算投組報酬。

如果 `sample_size >= env.num_stocks`，程式就會直接用全部股票。

### 為什麼每個 epoch 都會重新生成資料

因為這個專案沒有固定的歷史資料集。市場資料是透過 `MarketEnvironment.get_daily_returns(...)` 動態模擬產生的，所以每次呼叫都會從同一組分布參數重新抽樣。

這代表：

- 每個 epoch 的資料都不同
- 每個 batch 的資料也不同
- 訓練不是在固定 train set 上重複掃描
- 而是在同一個模擬分布上持續抽新樣本做近似最佳化

### loss 是怎麼計算的

單次模擬時，模型先產生 `portfolio_weights`，再用：

`portfolio_daily_returns = sampled_returns.T @ portfolio_weights`

算出該投組在整段期間內的每日報酬。之後把這條日報酬序列送入指定的 `loss_function`，得到一個 scalar loss。

程式裡做了：

`loss = loss_function(portfolio_daily_returns) / batch_size`

這樣做的目的是讓一個 epoch 內累積多次 `backward()` 後，梯度規模大致對應平均 loss，而不是總和。

### optimizer 是怎麼更新的

這份程式使用的是：

- `torch.optim.Adam(model.parameters(), lr=learning_rate)`

更新時機是：

- 每個 epoch 開始時先 `zero_grad()`
- 每個 batch loss 都 `backward()`
- 全部 batch 跑完後做一次 `optimizer.step()`

因此，模型參數是一個 epoch 更新一次，而不是每個 batch 更新一次。

### early stopping 是怎麼判斷的

程式用 `avg_loss_in_batch` 與 `best_loss` 做比較。

判斷規則是：

- 如果 `best_loss - avg_loss_in_batch > min_delta`
  - 視為有進步
  - 更新 `best_loss`
  - `patience_counter` 重設為 `patience`
- 否則
  - `patience_counter -= 1`
- 當 `patience_counter <= 0` 時停止訓練

### 這個模型是不是每天重新調倉

不是。

這份程式的模型**不會每天重新根據市場狀態做決策**。在單次訓練樣本中，模型只會根據一組股票 embedding 輸出一次固定權重，然後用這組固定權重去乘上一整段 `trading_days` 的每日報酬序列。

也就是說，這個專案的邏輯是：

- 先產生一組靜態權重
- 再觀察這組權重在一段時間內的累積表現

而不是：

- 每一天觀察新狀態
- 每一天重新決定買賣

這點非常重要，因為它決定了這個專案本質上是**靜態投組配置模型**，不是動態調倉策略。

## 8. 訓練與評估的差異

這個專案的訓練與評估不是使用固定切分的真實歷史資料，而是兩者都來自同一套模擬分布，但使用的是**不同次隨機抽樣出來的資料**。

### 訓練時使用多少天資料

訓練時，每次呼叫：

`env.get_daily_returns(env.trading_days)`

因此訓練期間長度是：

- `trading_days`

依照目前預設設定，`trading_days = 252`，也就是大約 1 年日資料。

### 評估時使用多少天資料

評估時，呼叫的是：

`env.get_daily_returns(env.trading_days * 3)`

因此評估期間長度是：

- `3 * trading_days`

依照目前預設設定，等於 `756` 天，大約 3 年。

### 訓練與評估是不是同一份資料

不是。

雖然訓練與評估都使用同一個 `MarketEnvironment` 內的分布參數，但每次 `get_daily_returns(...)` 都會重新隨機抽樣，所以：

- 訓練資料不是固定的
- 評估資料也不是固定的
- 評估時用到的那份資料，不是訓練時看過的同一批數值

### 這份專案是不是固定 4 年歷史資料切 train/test

不是。

它沒有固定歷史期間，也沒有「前幾年 train、後幾年 test」這種資料切分。這份專案的訓練與評估方式更接近：

- 在同一個模擬市場分布下持續抽樣
- 用較短期間資料訓練
- 用另一份較長期間資料評估

### 是否每次都重新模擬抽樣

是。

這是本專案很關鍵的特性。無論訓練或評估，資料都不是從硬碟讀出固定資料，而是每次執行時重新抽樣產生。

## 9. loss functions 解釋

本專案中的每個 loss function 都代表一種不同的投組優化目標。程式會針對每個 loss **分開訓練一個模型**，而不是把所有 loss 合成同一個總目標。

換句話說：

- 一個 loss 對應一個模型
- 一個模型對應一組權重
- 最後比較不同 loss 下的結果

下面逐一說明各 loss 的意義。

### 最大化收益：`loss_return`

- 公式：`-mean(p)`
- 目的：最大化平均報酬
- 特性：不直接懲罰波動或下行風險

### 最大化夏普：`loss_sharpe`

- 公式：`-(mean(p) / std(p))`
- 目的：最大化風險調整後報酬
- 特性：同時考慮報酬與總波動

### 最大化索提諾：`loss_sortino`

- 公式：`-(mean / downside_std)`
- 目的：最大化只考慮下行風險的風險調整後報酬
- 特性：只懲罰負報酬波動

### 最小化最大回撤：`loss_mdd`

- 公式：由累積報酬計算 drawdown，再取最小值
- 目的：降低投組在評估期間曾經出現的最大跌幅
- 注意：這個函式直接回傳 drawdown 的最小值，數值通常為負

### 最小化 CVaR：`loss_cvar`

- 公式：取 5% 最差報酬的平均值
- 目的：降低尾部風險
- 注意：在 `loss_functions` 字典中，實際使用的是 `lambda p: -loss_cvar(p)`，也就是用相反號來讓優化方向符合「最小化 loss」的訓練框架

### 收益 - 波動：`loss_return_vol`

- 公式：`-(mean - lam * std)`
- 目的：在報酬與波動之間做折衷
- 參數：`lam` 預設為 `0.5`

### 收益 - CVaR：`loss_return_cvar`

- 公式：`-(mean + lam * loss_cvar(p))`
- 目的：同時追求報酬並抑制尾部風險
- 注意：由於 `loss_cvar(p)` 通常是負值，這個式子實際上會把壞尾部結果納入懲罰

### 夏普 + 索提諾：`loss_sharpe_sortino`

- 公式：`a * loss_sharpe(p) + (1 - a) * loss_sortino(p)`
- 目的：結合總波動風險與下行風險的考量
- 參數：`a` 預設為 `0.5`

### 容易誤解的地方

這裡最容易誤解的地方有兩個：

1. 這些 loss 不是同時加總成一個總 loss。
2. 每個 loss 都會重新訓練一個新的模型。

因此不同 loss 的輸出結果，是不同模型之間的比較，而不是同一個模型在多目標損失下的結果。

## 10. 輸出結果說明

本專案執行完成後，會把結果存到：

- `result/results_時間戳/`

例如：

- `result/results_20260310-152952/`

### `performance_report.txt`

這是一份純文字績效報告，會記錄：

- 本次實驗名稱
- 每個 loss function 的績效指標
  - Annualized Return
  - Sharpe Ratio
  - Sortino Ratio
  - Max Drawdown
  - CVaR (5%)
  - Beta
  - IR

### 權重分布圖 `weights_*.png`

每個 loss 都會產生一張權重圖，檔名格式為：

- `weights_{loss名稱}_{trading_days}d_{sample_size}s.png`

例如：

- `weights_最大化夏普比率_252d_20s.png`

圖中會顯示：

- 橫軸：股票 ID
- 縱軸：投資權重
- 並用垂直線標出三個波動率 regime 的分界位置

### 累積收益圖 `cumulative_returns_*.png`

所有 loss 的累積收益曲線會畫在同一張圖上，檔名格式為：

- `cumulative_returns_{trading_days}d_{sample_size}s.png`

例如：

- `cumulative_returns_252d_20s.png`

圖例中會附上每個 loss 對應的年化報酬。

### `result/` 與 `results_時間戳/` 的用途

- `result/`
  - 作為所有執行結果的總資料夾。

- `results_時間戳/`
  - 作為某一次執行的獨立快照。
  - 可以避免不同執行結果互相覆蓋。

## 11. 專案限制與注意事項

以下限制與注意事項都直接來自目前程式碼的實際實作。

### 1. 不是使用真實市場資料

所有資料都來自模擬分布，不是從真實歷史價格、因子資料或基本面資料讀入。

### 2. `stock_embeddings` 是隨機生成的

`stock_embeddings = torch.randn(self.num_stocks, self.d_model)`，代表股票特徵本身沒有對應真實金融意義。模型實際上是在學習一組從隨機特徵到權重的映射。

### 3. 報酬是分布抽樣，不是真實市場時序

每次 `get_daily_returns(...)` 都是用常態分布重新抽樣，因此資料不具備真實市場常見的時序結構，例如：

- 波動叢聚
- regime switching
- fat tail
- serial correlation
- 真實橫截面關聯結構

### 4. 模型是靜態配置，不是動態交易

模型不會每天重新根據新資訊調整權重。一次 forward 產生一組固定權重後，就用這組權重套用到整段報酬期間。

### 5. `IR` 不是獨立計算

在 `PerformanceMetrics.calculate_metrics(...)` 中，`IR` 目前直接等於 `Sharpe Ratio`，不是以超額報酬相對 benchmark tracking error 計算的真正資訊比率。

### 6. `Beta` 的 benchmark 是簡化版 market proxy

`Beta` 不是對真實市場指數計算，而是以所有股票的平均日報酬作為 market returns。

### 7. 訓練與評估都來自同一個模擬分布

雖然數值樣本不同，但訓練與評估的分布假設相同。這代表評估結果比較偏向「在相同模擬假設下的泛化能力」，不等同於真實 out-of-sample 表現。

### 8. `batch_size` 的意義容易誤解

這裡的 `batch_size` 不是固定資料集上的 mini-batch 大小，而是每個 epoch 內重複抽樣模擬的次數。

### 9. 目前程式中的部分中文字串可能有編碼問題

從原始檔可見，某些註解與輸出文字有亂碼現象。這不影響主要訓練與評估邏輯，但會影響可讀性與終端機輸出內容。

## 12. 如何執行

### 入口檔案

專案入口是：

- `Toy_Model_Loss_Simulation.py`

### 如何修改 `config`

請直接修改 `Toy_Model_Loss_Simulation.py` 底部的 `config`：

```python
config = {
    'trading_days_per_epoch': 252,
    'learning_rate': 3e-5,
    'max_epochs': 100,
    'patience': 100,
    'min_delta': 1e-20,
    'batch_size': 10,
    'sample_size': 20,
}
```

你可以調整的重點包括：

- `trading_days_per_epoch`
- `learning_rate`
- `max_epochs`
- `patience`
- `min_delta`
- `batch_size`
- `sample_size`

### 執行方式

在專案根目錄下執行：

```bash
python Toy_Model_Loss_Simulation.py
```

### 執行後去哪裡看結果

程式執行後會在 `result/` 底下建立新的時間戳資料夾，例如：

```text
result/
  results_20260310-152952/
```

你可以在該資料夾中查看：

- `performance_report.txt`
- `weights_*.png`
- `cumulative_returns_*.png`

## 補充說明

如果要用一句話總結這個專案，可以這樣描述：

> 這是一個用人工模擬市場資料來測試不同投組 loss function 的 PyTorch toy portfolio optimization 專案；模型輸入股票 embedding，輸出靜態投資權重，並用模擬日報酬來比較不同風險報酬目標下的結果。

閱讀這份 README 時，最重要的三個觀念是：

1. 這不是用真實市場資料訓練的系統。
2. 這不是日頻動態調倉策略，而是靜態投組配置。
3. 每個 loss 都是各自獨立訓練一個模型，再比較結果。

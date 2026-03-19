import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time
from tqdm import tqdm
from matplotlib.font_manager import FontProperties #! <--- ADDED: 导入字体管理器

from market_environment import MarketEnvironment
from portfolio_model import PortfolioModel
from performance_metrics import PerformanceMetrics

# --- 0. 初始化设置 ---

#! <--- MODIFIED: 动态加载字体文件 ---!
# 检查字体文件是否存在于脚本的同一目录
base_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(base_dir, 'SourceHanSansSC-Regular.otf')

if os.path.exists(font_path):
    chinese_font = FontProperties(fname=font_path)
    print(f"成功加载中文字体: {font_path}")
else:
    chinese_font = None
    print(f"警告: 未在脚本目录中找到字体 '{font_path}'。")
    print("图表中的中文可能无法正常显示。请下载该字体并放置于脚本同级目录。")


# --- 损失函数定义 (无变化) ---
def loss_return(p): return -torch.mean(p)
def loss_sharpe(p): return -(torch.mean(p) / (torch.std(p) + 1e-8))
def loss_sortino(p):
    mean_return = torch.mean(p)
    downside = p[p < 0]
    if len(downside) == 0: return -mean_return
    return -(mean_return / (torch.std(downside) + 1e-8))
def loss_mdd(p):
    cum_ret = torch.cumprod(1 + p, dim=0)
    peak, _ = torch.cummax(cum_ret, dim=0)
    drawdown = (cum_ret - peak) / peak
    return torch.min(drawdown)
def loss_cvar(p):
    q = 0.05
    percentile = torch.quantile(p, q)
    return torch.mean(p[p <= percentile])
def loss_return_vol(p, lam=0.5): return -(torch.mean(p) - lam * torch.std(p))
def loss_return_cvar(p, lam=0.5): return -(torch.mean(p) + lam * loss_cvar(p))
def loss_sharpe_sortino(p, a=0.5): return a * loss_sharpe(p) + (1 - a) * loss_sortino(p)


# --- 4. 训练与评估函数 (已重构以支持采样和评估) ---
def train_and_evaluate(loss_function, loss_name, env, learning_rate, device,
                       sample_size, batch_size, max_epochs, patience, min_delta):
    """
    完整的训练和评估流程，支持采样、梯度累积和早停。
    """
    print(f"--- 开始训练: {loss_name} | Epoch长度: {env.trading_days}天 | 样本数: {sample_size} | Batch Size: {batch_size} ---")
    
    # 模型的输入维度是d_model，与股票总数无关
    model = PortfolioModel(env.d_model).to(device)
    stock_embeddings = env.stock_embeddings.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_loss = float('inf')
    patience_counter = patience

    progress_bar = tqdm(range(max_epochs), desc=f"Training {loss_name}", unit="epoch")
    completed_epochs = 0

    for epoch in progress_bar:
        epoch_start_time = time.time()
        optimizer.zero_grad()
        total_loss_in_batch = 0.0

        for _ in range(batch_size):
            # --- 采样逻辑 ---
            all_daily_returns = env.get_daily_returns(env.trading_days).to(device)
            # 从所有股票中随机抽取 'sample_size' 个索引
            if sample_size < env.num_stocks:
                indices = torch.randperm(env.num_stocks, device=device)[:sample_size]
            else: # 如果sample_size等于总数，则不进行采样
                indices = torch.arange(env.num_stocks, device=device)
            
            sampled_embeddings = stock_embeddings[indices]
            sampled_returns = all_daily_returns[indices, :]
            
            # --- 前向/反向传播 ---
            portfolio_weights = model(sampled_embeddings) # 模型只在样本上计算权重
            portfolio_daily_returns = torch.matmul(sampled_returns.T, portfolio_weights)
            
            loss = loss_function(portfolio_daily_returns) / batch_size
            loss.backward()
            total_loss_in_batch += loss.item() * batch_size

        optimizer.step()
        
        avg_loss_in_batch = total_loss_in_batch / batch_size
        epoch_elapsed_time = time.time() - epoch_start_time
        progress_bar.set_postfix(loss=f"{avg_loss_in_batch:.6f}", epoch_time=f"{epoch_elapsed_time:.2f}s")
        if (epoch + 1) % 30 == 0: # 每30个epoch打印一次日志
            print(f"[{loss_name}] Epoch {epoch+1}/{max_epochs} finished in {epoch_elapsed_time:.2f}s, loss={avg_loss_in_batch:.6f}")
        completed_epochs = epoch + 1
            
        # --- 早停逻辑 ---
        if best_loss - avg_loss_in_batch > min_delta:
            best_loss = avg_loss_in_batch
            patience_counter = patience
        else:
            patience_counter -= 1
        if patience_counter <= 0:
            print(f"早停机制触发于 Epoch {epoch+1}。")
            break

    progress_bar.close()

    print(f"--- {loss_name} 训练完成 (共 {completed_epochs} 个 epochs), 开始评估 ---")
    
    # --- 评估逻辑 (在所有股票上评估，以获得全局表现) ---
    with torch.no_grad():
        # 获取所有股票的权重
        full_weights = model(stock_embeddings).detach()
        eval_returns_stocks = env.get_daily_returns(env.trading_days * 3).to(device)
        portfolio_eval_returns = torch.matmul(eval_returns_stocks.T, full_weights)
    
    metrics_calculator = PerformanceMetrics(trading_days=env.trading_days)
    performance = metrics_calculator.calculate_metrics(portfolio_eval_returns, eval_returns_stocks)
    
    cumulative_returns = torch.cumprod(1 + portfolio_eval_returns, dim=0)
    return full_weights.cpu().numpy(), cumulative_returns.cpu().numpy(), performance


# --- 5. 实验运行与结果输出 (已修改绘图部分) ---
def run_experiment(trading_days_per_epoch, learning_rate, max_epochs, patience, min_delta,
                   device, batch_size, sample_size, output_dir):
    exp_name = f"Epoch长度={trading_days_per_epoch}天_样本数={sample_size}"
    print(f"\n{'=' * 35}\n 开始新一轮实验: {exp_name} \n{'=' * 35}\n")
    env = MarketEnvironment(trading_days=trading_days_per_epoch)
    if sample_size > env.num_stocks:
        print(f"警告: 样本数 ({sample_size}) 大于股票总数 ({env.num_stocks})。将使用全部股票。")
        sample_size = env.num_stocks

    loss_functions = {
        "最大化收益": loss_return, "最大化夏普比率": loss_sharpe, "最大化索提诺比率": loss_sortino,
        "最小化最大回撤": loss_mdd, "最小化CVaR": lambda p: -loss_cvar(p), "收益-波动": loss_return_vol,
        "收益-CVaR": loss_return_cvar, "夏普+索提诺": loss_sharpe_sortino,
    }

    '''
    loss_functions = {
        "最大化夏普比率": loss_sharpe, "收益-波动": loss_return_vol,
        "收益-CVaR": loss_return_cvar,
    }'''

    results = {}
    report_path = os.path.join(output_dir, 'performance_report.txt')
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'=' * 30}\n实验设置: {exp_name}\n{'=' * 30}\n")

    for name, loss_func in loss_functions.items():
        weights, cumulative_returns, performance = train_and_evaluate(
            loss_func,
            name,
            env,
            learning_rate,
            device,
            sample_size,
            batch_size,
            max_epochs,
            patience,
            min_delta
        )
        results[name] = {'weights': weights, 'cumulative_returns': cumulative_returns, 'performance': performance}
        print("\n最终表现指标:")
        for key, value in performance.items(): print(f"{key}: {value:.4f}")
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- 损失函数: {name} ---\n")
            for key, value in performance.items(): f.write(f"{key}: {value:.4f}\n")
            f.write("-" * 25 + "\n")

        # --- MODIFIED: 保存权重分布图 (使用加载的字体) ---
        safe_name = name.replace(' ', '_').replace('+', 'and')
        fig_weights, ax = plt.subplots(1, 1, figsize=(12, 5))
        stock_ids = np.arange(env.num_stocks)
        ax.bar(stock_ids, weights, width=5.0)
        # 在所有需要中文的地方，传入 fontproperties 参数
        ax.set_title(f'投资权重: {name}\n({exp_name})', fontproperties=chinese_font)
        ax.set_xlabel('股票ID (0-2000: 5% vol, 2001-4100: 10% vol, 4101-6201: 15% vol)', fontproperties=chinese_font)
        ax.set_ylabel('投资权重', fontproperties=chinese_font)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xlim([0, env.num_stocks])
        ax.axvline(x=2000.5, color='r', linestyle='--', label='5% vs 10% vol')
        ax.axvline(x=4100.5, color='g', linestyle='--', label='10% vs 15% vol')
        ax.legend()
        filename = f"weights_{safe_name}_{trading_days_per_epoch}d_{sample_size}s.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close(fig_weights)

    # --- MODIFIED: 保存累积收益对比图 (使用加载的字体) ---
    fig_cum, ax_cum = plt.subplots(1, 1, figsize=(14, 8))
    for name, result in results.items():
        label_text = f"{name} (年化收益: {result['performance']['Annualized Return']:.2%})"
        ax_cum.plot(result['cumulative_returns'], label=label_text)

    ax_cum.set_title(f'不同策略的长期累积收益对比\n({exp_name})', fontproperties=chinese_font)
    ax_cum.set_xlabel('交易日 (3年)', fontproperties=chinese_font)
    ax_cum.set_ylabel('累积收益', fontproperties=chinese_font)
    # 为图例也设置字体
    ax_cum.legend(loc='upper left', prop=chinese_font)
    ax_cum.grid(True)
    filename_cum = f"cumulative_returns_{trading_days_per_epoch}d_{sample_size}s.png"
    plt.savefig(os.path.join(output_dir, filename_cum))
    plt.close(fig_cum)
    print(f"\n--- 实验 {exp_name} 完成！所有结果已保存至目录: {output_dir} ---\n")


def main(config):
    # --- 以目前腳本所在目錄為基準 ---
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # --- 建立 result 資料夾 ---
    result_dir = os.path.join(base_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    # --- 在 result 底下建立本次實驗輸出目錄 ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(result_dir, f"results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # --- 设置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*20} 使用设备: {device} {'='*20}")
    print(f"{'='*20} 所有结果将保存在: {output_dir} {'='*20}")

    run_experiment(
        trading_days_per_epoch=config['trading_days_per_epoch'],
        learning_rate=config['learning_rate'],
        max_epochs=config['max_epochs'],
        patience=config['patience'],
        min_delta=config['min_delta'],
        device=device,
        batch_size=config['batch_size'],
        sample_size=config['sample_size'],
        output_dir=output_dir
    )


if __name__ == '__main__':
    config = {
        'trading_days_per_epoch': 252,
        'learning_rate': 3e-5,
        'max_epochs': 100,
        'patience': 100,
        'min_delta': 1e-20,
        'batch_size': 10,
        'sample_size': 20,
    }

    main(config)

"""
EchoMarsh 模拟实盘回测引擎 (Paper Trading Backtest)
---------------------------------------------------
用训练好的模型在历史数据上进行逐日模拟交易，
严格遵守 A 股规则（T+1, 涨停买不到, 集合竞价）。

用法：
    python scripts/paper_trade.py --start 20240101 --end 20241231
"""

import os
import sys
import argparse
import datetime
import glob

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from models.backbone.model_factory import ModelFactory
from core.scanner.preprocessor import Preprocessor


INITIAL_CAPITAL = 20_000.0  # 初始资金 2 万元
COMMISSION_RATE = 0.0003    # 双边手续费 0.03%（实际可根据券商调整）
STAMP_TAX       = 0.001     # 印花税 0.1%（仅卖出时收取）
SLIPPAGE        = 0.002     # 每次交易的滑点假设 0.2%


def load_model(checkpoint_path, device):
    model, _ = ModelFactory.create_model(model_type='transformer', ts_feature_dim=22)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model


def score_stock(model, file_path, preprocessor, feature_cols, device, seq_len=60):
    """给单只股票的今日 CSV 打分，返回早盘综合涨停概率"""
    df = preprocessor.process_file(file_path)
    if df is None or len(df) < seq_len:
        return 0.0

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return 0.0

    features = df[feature_cols].values.astype(np.float32)

    # 只用早盘前 30 分钟（09:30 ~ 10:00）的样本综合打分
    scores = []
    with torch.no_grad():
        for i in range(min(30, len(features) - seq_len + 1)):
            x   = torch.tensor(features[i:i+seq_len]).unsqueeze(0).to(device)
            meta= torch.zeros(1, 7).to(device)
            out = model(x, meta)
            scores.append(torch.sigmoid(out[0, 1]).item())
    return float(np.mean(scores)) if scores else 0.0


def get_actual_return(file_path):
    """获取股票当天的实际收益率（收盘价 vs 开盘价）"""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return 0.0
        # 以当天第一根分钟 K 线开盘价买入，收盘卖出
        buy_price  = df['开盘'].iloc[0]
        sell_price = df['收盘'].iloc[-1]
        return (sell_price - buy_price) / buy_price
    except Exception:
        return 0.0


def run_paper_trade(data_dir, checkpoint_path, start_str, end_str, top_n=3):
    """
    逐日模拟回测。每天选模型评分最高的 top_n 只股票，平均仓位买入，收盘全部卖出。
    """
    start_dt = datetime.datetime.strptime(start_str, '%Y%m%d')
    end_dt   = datetime.datetime.strptime(end_str,   '%Y%m%d')

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model    = load_model(checkpoint_path, device)
    preprocessor  = Preprocessor()
    feature_cols  = [f'{c}_norm' for c in preprocessor.feature_cols]

    # 回测状态
    capital        = INITIAL_CAPITAL
    equity_curve   = [(start_dt, capital)]
    trade_log      = []
    win_trades     = 0
    total_trades   = 0

    current_dt = start_dt
    while current_dt <= end_dt:
        date_str  = current_dt.strftime('%Y%m%d')
        day_files = glob.glob(os.path.join(data_dir, f"*_{date_str}.csv"))

        if day_files:
            # 给今日所有股票打分
            candidates = []
            for f in day_files:
                symbol = os.path.basename(f).split('_')[0]
                score  = score_stock(model, f, preprocessor, feature_cols, device)
                candidates.append((symbol, score, f))

            candidates.sort(key=lambda x: x[1], reverse=True)
            selected = candidates[:top_n]

            # 模拟交易
            per_stock_capital = capital / len(selected) if selected else 0
            day_pnl = 0.0

            for symbol, score, f in selected:
                actual_ret = get_actual_return(f)
                
                # 成本 = 买入手续费 + 滑点
                cost = per_stock_capital * (COMMISSION_RATE + SLIPPAGE)
                # 收益（卖出时还需要印花税）
                gross_pnl = per_stock_capital * actual_ret
                sell_cost = per_stock_capital * (COMMISSION_RATE + STAMP_TAX)
                net_pnl   = gross_pnl - cost - sell_cost

                day_pnl += net_pnl
                total_trades += 1
                if actual_ret > 0:
                    win_trades += 1

                trade_log.append({
                    'date': date_str,
                    'symbol': symbol,
                    'model_score': score,
                    'actual_return': actual_ret,
                    'net_pnl': net_pnl,
                })

            capital += day_pnl

        equity_curve.append((current_dt, capital))
        current_dt += datetime.timedelta(days=1)

    # ===================== 计算绩效指标 =====================
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate     = win_trades / total_trades * 100 if total_trades else 0
    
    equity_vals = [e[1] for e in equity_curve]
    rolling_max = np.maximum.accumulate(equity_vals)
    drawdowns   = (np.array(equity_vals) - rolling_max) / (rolling_max + 1e-8)
    max_drawdown= float(np.min(drawdowns)) * 100

    # 日收益率用于夏普
    daily_rets = np.diff(equity_vals) / (np.array(equity_vals[:-1]) + 1e-8)
    sharpe = (np.mean(daily_rets) / (np.std(daily_rets) + 1e-8)) * np.sqrt(252) if len(daily_rets) > 1 else 0

    print("\n" + "=" * 50)
    print("EchoMarsh Paper Trading Report (模拟回测)")
    print("=" * 50)
    print(f"回测区间: {start_str} ~ {end_str}")
    print(f"初始资金: {INITIAL_CAPITAL:,.0f} 元")
    print(f"最终资金: {capital:,.1f} 元")
    print(f"总收益率: {total_return:+.2f}%")
    print(f"年化夏普: {sharpe:.2f}")
    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"交易次数: {total_trades}")
    print(f"胜率:     {win_rate:.1f}%")
    print("=" * 50)

    # ===================== 生成资金曲线图 =====================
    report_dir = os.path.join(project_root, 'logs', 'backtest')
    os.makedirs(report_dir, exist_ok=True)

    dates  = [e[0] for e in equity_curve]
    values = [e[1] for e in equity_curve]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f"EchoMarsh 模拟回测报告 | {start_str} ~ {end_str}", fontsize=14, fontweight='bold')

    ax1 = axes[0]
    ax1.plot(dates, values, color='#2196F3', linewidth=2, label='EchoMarsh 资金曲线')
    ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.6, label='初始资金')
    ax1.fill_between(dates, INITIAL_CAPITAL, values,
                     where=[v >= INITIAL_CAPITAL for v in values], alpha=0.2, color='green')
    ax1.fill_between(dates, INITIAL_CAPITAL, values,
                     where=[v < INITIAL_CAPITAL for v in values], alpha=0.2, color='red')
    ax1.set_ylabel('账户净值 (元)')
    ax1.set_title(f"总收益: {total_return:+.1f}% | 夏普: {sharpe:.2f} | 最大回撤: {max_drawdown:.1f}%")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 每日 PnL 柱状图
    if trade_log:
        df_log = pd.DataFrame(trade_log)
        daily_pnl = df_log.groupby('date')['net_pnl'].sum()
        ax2 = axes[1]
        colors = ['#4CAF50' if v >= 0 else '#F44336' for v in daily_pnl.values]
        ax2.bar(range(len(daily_pnl)), daily_pnl.values, color=colors, alpha=0.8)
        ax2.set_ylabel('每日盈亏 (元)')
        ax2.set_title(f"每日盈亏 | 胜率: {win_rate:.1f}% | 总交易: {total_trades} 次")
        ax2.axhline(y=0, color='black', linewidth=0.8)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(report_dir, f"paper_trade_{start_str}_{end_str}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Report] 资金曲线图已保存: {plot_path}")

    # 保存交易日志
    if trade_log:
        log_path = os.path.join(report_dir, f"trade_log_{start_str}_{end_str}.csv")
        pd.DataFrame(trade_log).to_csv(log_path, index=False, encoding='utf-8-sig')
        print(f"[Report] 交易日志已保存: {log_path}")

    return capital, total_return, sharpe, max_drawdown, win_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start',      default='20240101')
    parser.add_argument('--end',        default=datetime.date.today().strftime('%Y%m%d'))
    parser.add_argument('--data_dir',   default=os.path.join(project_root, 'data', 'raw'))
    parser.add_argument('--checkpoint', default=os.path.join(project_root, 'models', 'checkpoints', 'best_echomarsh_model.pth'))
    parser.add_argument('--top_n',      type=int, default=3)
    args = parser.parse_args()

    run_paper_trade(args.data_dir, args.checkpoint, args.start, args.end, args.top_n)

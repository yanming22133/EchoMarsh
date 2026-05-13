"""
EchoMarsh 每日复盘与可视化仪表板 (Daily Review Dashboard)
---------------------------------------------------------
收盘后（15:30 以后）运行此脚本，系统将自动完成：
1. 加载今日行情数据（需已通过 data_fetcher 保存到 data/raw/）
2. 模型推理：给今日扫描到的目标打分
3. 生成今日可视化报告（资金曲线、信号准确性）
4. 在线 Fine-tuning：用今日真实结果微调模型
5. 将报告保存到 logs/daily_review/YYYYMMDD/

用法：
    python scripts/daily_review.py --date 20240523
"""

import os
import sys
import argparse
import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # 无显示器的服务器模式
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块

from models.backbone.model_factory import ModelFactory
from core.scanner.preprocessor import Preprocessor
from core.trainer.trainer import EchoMarshTrainer


def load_model_for_inference(checkpoint_path, device):
    model, _ = ModelFactory.create_model(model_type='transformer', ts_feature_dim=22)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[WARN] Checkpoint not found: {checkpoint_path}. Using random weights.")
    model.to(device)
    model.eval()
    return model


def run_inference_on_file(model, file_path, device, preprocessor, feature_cols):
    """对单只股票的 CSV 文件进行推理，返回逐分钟的预测值"""
    df = preprocessor.process_file(file_path)
    if df is None:
        return None, None

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return None, None

    features  = df[feature_cols].values.astype(np.float32)
    true_ret  = df['target_max_return_15m'].values.astype(np.float32)
    seq_len   = 60

    preds_ret = []
    preds_cls = []

    with torch.no_grad():
        for i in range(len(features) - seq_len + 1):
            x    = torch.tensor(features[i:i+seq_len], dtype=torch.float32).unsqueeze(0).to(device)
            meta = torch.zeros(1, 7).to(device)
            out  = model(x, meta)
            preds_ret.append(out[0, 0].item())
            preds_cls.append(torch.sigmoid(out[0, 1]).item())  # 转化为概率

    return preds_ret, preds_cls, df, true_ret


def generate_daily_report(date_str, data_dir, checkpoint_path, report_dir):
    """生成每日复盘报告与可视化"""
    import glob
    os.makedirs(report_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model_for_inference(checkpoint_path, device)

    preprocessor  = Preprocessor()
    feature_cols  = [f'{c}_norm' for c in preprocessor.feature_cols]

    # 获取今日所有文件
    today_files = glob.glob(os.path.join(data_dir, f"*_{date_str}.csv"))
    if not today_files:
        print(f"[INFO] 今日 ({date_str}) 无数据文件，请先运行 data_fetcher。")
        return

    stock_results = []
    for f in today_files:
        symbol = os.path.basename(f).split('_')[0]
        result = run_inference_on_file(model, f, device, preprocessor, feature_cols)
        if result[0] is None:
            continue
        preds_ret, preds_cls, df, true_ret = result
        
        # 重点关注早盘前 60 分钟的信号
        early_session_cls = np.mean(preds_cls[:30]) if preds_cls else 0  # 前30分钟平均涨停概率
        actual_max_ret    = float(np.nanmax(true_ret[:30])) if len(true_ret) >= 30 else 0
        
        stock_results.append({
            'symbol': symbol,
            'file': f,
            'early_limitup_prob': early_session_cls,
            'actual_max_return': actual_max_ret,
            'preds_ret': preds_ret,
            'preds_cls': preds_cls,
            'df': df,
        })

    if not stock_results:
        print("今日无有效推理结果。")
        return

    # 按模型置信度排序
    stock_results.sort(key=lambda x: x['early_limitup_prob'], reverse=True)

    # ===================== 可视化生成 =====================
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle(f"EchoMarsh 每日复盘报告 | {date_str}", fontsize=16, fontweight='bold')

    # 图1：今日推荐 vs 实际表现（胜率图）
    ax1 = axes[0]
    symbols    = [r['symbol'] for r in stock_results[:10]]
    model_scores = [r['early_limitup_prob'] for r in stock_results[:10]]
    actual_rets  = [r['actual_max_return'] * 100 for r in stock_results[:10]]
    
    x = np.arange(len(symbols))
    w = 0.35
    bars1 = ax1.bar(x - w/2, model_scores, w, label='模型涨停概率 (%)', color='#2196F3', alpha=0.8)
    bars2 = ax1.bar(x + w/2, actual_rets, w, label='实际最大涨幅 (%)', color='#4CAF50', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(symbols, rotation=30)
    ax1.set_ylabel('百分比 (%)')
    ax1.set_title('今日推荐股票：模型评分 vs 实际表现（取前10名）')
    ax1.legend()
    ax1.axhline(y=9.5, color='red', linestyle='--', alpha=0.5, label='涨停线 9.5%')
    ax1.grid(True, alpha=0.3)

    # 图2：模型置信度 vs 真实涨幅的散点图（评估信号质量）
    ax2 = axes[1]
    all_scores = [r['early_limitup_prob'] for r in stock_results]
    all_actual = [r['actual_max_return'] * 100 for r in stock_results]
    ax2.scatter(all_scores, all_actual, alpha=0.6, color='#9C27B0', edgecolors='white', s=60)
    ax2.set_xlabel('模型涨停概率')
    ax2.set_ylabel('实际涨幅 (%)')
    ax2.set_title('信号质量散点图（越右上方聚集，信号越精准）')
    ax2.axhline(y=9.5, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # 图3：最高置信度票的分时预测曲线
    ax3 = axes[2]
    top_stock = stock_results[0]
    if top_stock['preds_cls']:
        time_axis = list(range(len(top_stock['preds_cls'])))
        ax3.plot(time_axis, [p * 100 for p in top_stock['preds_cls']],
                 color='#2196F3', linewidth=2, label='预测涨停概率 (%)')
        if len(top_stock['df']) > 0:
            actual_curve = top_stock['df']['收盘'].pct_change().fillna(0).cumsum() * 100
            ax3.plot(actual_curve.values[:len(time_axis)],
                     color='#4CAF50', linewidth=1.5, linestyle='--', label='实际累计涨幅 (%)')
        ax3.set_xlabel('分钟')
        ax3.set_ylabel('百分比 (%)')
        ax3.set_title(f"最高评分个股分时分析：{top_stock['symbol']}")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    report_path = os.path.join(report_dir, f"daily_review_{date_str}.png")
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Report] 可视化报告已保存: {report_path}")

    # ===================== 文字复盘摘要 =====================
    n_correct = sum(1 for r in stock_results if r['actual_max_return'] > 0.05)
    hit_rate  = n_correct / len(stock_results) * 100 if stock_results else 0
    
    summary_path = os.path.join(report_dir, f"summary_{date_str}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f_out:
        f_out.write(f"EchoMarsh 每日复盘摘要 | {date_str}\n")
        f_out.write("=" * 50 + "\n")
        f_out.write(f"扫描股票数量: {len(stock_results)}\n")
        f_out.write(f"涨幅超过5%的数量: {n_correct}\n")
        f_out.write(f"大涨命中率: {hit_rate:.1f}%\n\n")
        f_out.write("今日推荐 Top5:\n")
        for i, r in enumerate(stock_results[:5]):
            f_out.write(f"  {i+1}. {r['symbol']} | 涨停概率: {r['early_limitup_prob']:.2%} | 实际涨幅: {r['actual_max_return']:.2%}\n")
    print(f"[Report] 文字摘要已保存: {summary_path}")

    return stock_results


def fine_tune_with_today(date_str, data_dir, checkpoint_path, fine_tune_epochs=3):
    """在线 Fine-tuning：用今日实际结果对模型进行轻量微调"""
    import glob
    from core.scanner.dataset import EchoMarshDatasetFromFiles
    from torch.utils.data import DataLoader

    today_files = glob.glob(os.path.join(data_dir, f"*_{date_str}.csv"))
    if not today_files:
        print("[FineTune] 今日无数据，跳过在线学习。")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model_for_inference(checkpoint_path, device)
    model.train()

    ds = EchoMarshDatasetFromFiles(today_files)
    if len(ds) == 0:
        print("[FineTune] 今日样本为空，跳过微调。")
        return

    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
    trainer = EchoMarshTrainer(
        model=model, device=device,
        checkpoint_dir=os.path.dirname(checkpoint_path),
        lr=1e-5,  # Fine-tune 用极小学习率，防止灾难性遗忘
        epochs=fine_tune_epochs,
        patience=fine_tune_epochs,
    )

    print(f"[FineTune] 在线学习：对 {len(ds)} 条今日样本进行 {fine_tune_epochs} 轮微调...")
    trainer.fit(loader)

    # 保存 fine-tuned 权重
    ft_path = checkpoint_path.replace('.pth', f'_finetune_{date_str}.pth')
    torch.save(model.state_dict(), ft_path)
    print(f"[FineTune] Fine-tuned 权重已保存: {ft_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoMarsh Daily Review")
    parser.add_argument('--date', default=datetime.date.today().strftime('%Y%m%d'))
    parser.add_argument('--data_dir',    default=os.path.join(project_root, 'data', 'raw'))
    parser.add_argument('--checkpoint',  default=os.path.join(project_root, 'models', 'checkpoints', 'best_echomarsh_model.pth'))
    parser.add_argument('--no_finetune', action='store_true', help='跳过在线Fine-tuning')
    args = parser.parse_args()

    review_dir = os.path.join(project_root, 'logs', 'daily_review', args.date)
    
    print(f"EchoMarsh 每日复盘 | {args.date}")
    generate_daily_report(args.date, args.data_dir, args.checkpoint, review_dir)

    if not args.no_finetune:
        fine_tune_with_today(args.date, args.data_dir, args.checkpoint)

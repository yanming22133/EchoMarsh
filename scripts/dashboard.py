"""
EchoMarsh 实盘可视化仪表板 (Live Terminal Dashboard)
---------------------------------------------------
启动后在终端展示实时运行状态，包括：
  - 系统状态 / 市场时段
  - 模型实时扫描信号（涨停候选榜）
  - 当前持仓状态
  - 今日已执行的交易记录
  - GPU / CPU 实时负载

用法:
    python scripts/dashboard.py --paper       (模拟模式)
    python scripts/dashboard.py --live        (实盘模式，需已连接 QMT)
"""

import os
import sys
import time
import glob
import random
import argparse
import threading
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich import box

from models.backbone.model_factory import ModelFactory
from core.scanner.preprocessor import Preprocessor
from core.executor.qmt_executor import QMTExecutor


console = Console()

# ============================================================
# 算法优化：推理引擎
# ============================================================
class FastInferenceEngine:
    """
    高速推理引擎 - 对算法的核心优化：
    1. torch.inference_mode() 代替 torch.no_grad() (更快)
    2. torch.cuda.amp.autocast() 混合精度推理 (FP16 on GPU)
    3. 批量推理 (一次性处理所有候选股)
    4. 模型编译 (torch.compile 如果 PyTorch 2.0+)
    """
    def __init__(self, checkpoint_path, device, ts_feature_dim=22):
        self.device = device
        self.preprocessor = Preprocessor()
        self.feature_cols = [f'{c}_norm' for c in self.preprocessor.feature_cols]
        self.seq_len = 60

        model, _ = ModelFactory.create_model(model_type='transformer', ts_feature_dim=ts_feature_dim)
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # 优化 1: 切换到 eval 模式并关闭 Dropout
        model.eval()

        # 优化 2: 如果 CUDA 可用，使用混合精度 (FP16 速度是 FP32 的 2x)
        self.use_amp = (device.type == 'cuda')

        # 优化 3: torch.compile (PyTorch 2.0+，编译后首批推理有延迟，之后极快)
        try:
            model = torch.compile(model)
            console.print("[green]torch.compile 优化已启用 (首次推理会预热 ~5s)[/]")
        except Exception:
            pass  # 旧版 PyTorch 不支持，忽略

        self.model = model.to(device)

    @torch.inference_mode()
    def batch_score(self, file_list: list) -> list:
        """
        批量高速推理：一次将所有候选股的早盘序列打包送入 GPU
        返回 [(symbol, limit_up_prob, return_pred), ...]
        """
        batch_ts = []
        batch_meta = []
        symbols = []

        for f in file_list:
            symbol = os.path.basename(f).split('_')[0]
            df = self.preprocessor.process_file(f)
            if df is None or len(df) < self.seq_len:
                continue
            missing = [c for c in self.feature_cols if c not in df.columns]
            if missing:
                continue

            features = df[self.feature_cols].values.astype(np.float32)
            # 取最新的一个窗口（对应盘中当前时刻）
            x = features[-self.seq_len:]
            if np.isnan(x).any() or np.isinf(x).any():
                continue

            batch_ts.append(x)
            batch_meta.append(np.zeros(7, dtype=np.float32))
            symbols.append(symbol)

        if not batch_ts:
            return []

        # 优化 4: 一次性打包为张量，避免 Python 循环中的重复 GPU 传输
        ts_tensor   = torch.tensor(np.array(batch_ts),   dtype=torch.float32).to(self.device)
        meta_tensor = torch.tensor(np.array(batch_meta), dtype=torch.float32).to(self.device)

        # 优化 5: 混合精度推理（FP16）
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(ts_tensor, meta_tensor)
        else:
            outputs = self.model(ts_tensor, meta_tensor)

        results = []
        for i, sym in enumerate(symbols):
            ret_pred   = outputs[i, 0].item()
            limit_prob = torch.sigmoid(outputs[i, 1]).item()
            results.append((sym, limit_prob, ret_pred))

        # 按涨停概率降序排列
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# ============================================================
# 可视化面板生成函数
# ============================================================
def make_header(market_status: str) -> Panel:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_color = {
        "集合竞价": "yellow", "盘中交易": "green",
        "午休": "blue", "已收盘": "red", "等待开盘": "cyan"
    }.get(market_status, "white")
    
    title = Text()
    title.append("⚡ EchoMarsh ", style="bold cyan")
    title.append("鸣泽量化系统", style="bold white")
    title.append("  |  ", style="dim")
    title.append(now, style="dim white")
    title.append("  |  市场状态: ", style="dim")
    title.append(market_status, style=f"bold {status_color}")
    return Panel(title, style="bold blue")


def make_signal_table(signals: list) -> Panel:
    table = Table(
        title="[bold yellow]⚡ 实时扫描信号（涨停候选榜）",
        box=box.ROUNDED, show_header=True, header_style="bold magenta"
    )
    table.add_column("排名", style="bold white", width=4)
    table.add_column("代码", style="cyan", width=10)
    table.add_column("涨停概率", width=12)
    table.add_column("收益预测", width=10)
    table.add_column("信号强度", width=20)

    for i, (sym, prob, ret) in enumerate(signals[:10]):
        prob_pct = prob * 100
        ret_pct = ret * 100
        
        # 颜色评级
        if prob_pct >= 80:
            prob_style = "bold red"
            bar = "█" * 10
        elif prob_pct >= 60:
            prob_style = "bold yellow"
            bar = "█" * 7 + "░" * 3
        elif prob_pct >= 40:
            prob_style = "green"
            bar = "█" * 5 + "░" * 5
        else:
            prob_style = "white"
            bar = "█" * 3 + "░" * 7

        ret_style = "green" if ret_pct >= 0 else "red"
        table.add_row(
            f"[bold white]{i+1}[/]",
            f"[cyan]{sym}[/]",
            f"[{prob_style}]{prob_pct:.1f}%[/]",
            f"[{ret_style}]{ret_pct:+.2f}%[/]",
            f"[{prob_style}]{bar}[/]",
        )

    return Panel(table, border_style="yellow")


def make_position_table(executor: QMTExecutor) -> Panel:
    table = Table(
        title="[bold green]💼 当前持仓",
        box=box.ROUNDED, show_header=True, header_style="bold green"
    )
    table.add_column("代码", style="cyan", width=10)
    table.add_column("持股数", width=8)
    table.add_column("成本价", width=10)
    table.add_column("当前价", width=10)
    table.add_column("浮盈亏", width=12)

    if not executor.positions:
        table.add_row("[dim]暂无持仓[/]", "-", "-", "-", "-")
    else:
        for pos in executor.positions:
            cur_price = executor._get_realtime_price(pos.symbol)
            pnl_pct = (cur_price - pos.cost_price) / pos.cost_price * 100 if pos.cost_price else 0
            pnl_style = "green" if pnl_pct >= 0 else "red"
            table.add_row(
                pos.symbol, str(pos.volume),
                f"{pos.cost_price:.2f}",
                f"{cur_price:.2f}",
                f"[{pnl_style}]{pnl_pct:+.2f}%[/]"
            )

    cash = executor.get_cash()
    total = sum(p.volume * executor._get_realtime_price(p.symbol) for p in executor.positions) + cash
    summary = Text(f"  现金: ¥{cash:,.1f}  |  账户总值: ¥{total:,.1f}  |  今日下单: {executor.risk.daily_order_count} 次", style="dim")
    return Panel(table, border_style="green", subtitle=summary)


def make_order_table(executor: QMTExecutor) -> Panel:
    table = Table(
        title="[bold blue]📋 今日交易记录",
        box=box.SIMPLE, show_header=True, header_style="bold blue"
    )
    table.add_column("时间", width=8)
    table.add_column("方向", width=6)
    table.add_column("代码", width=10)
    table.add_column("价格", width=8)
    table.add_column("数量", width=8)
    table.add_column("状态", width=10)

    recent = executor.order_history[-8:]
    if not recent:
        table.add_row("[dim]暂无交易[/]", "-", "-", "-", "-", "-")
    else:
        for o in reversed(recent):
            d_style = "green" if o.direction == 'BUY' else "red"
            s_style = "green" if o.status == 'FILLED' else "yellow"
            table.add_row(
                o.timestamp, f"[{d_style}]{o.direction}[/]",
                o.symbol, f"{o.price:.2f}", str(o.volume),
                f"[{s_style}]{o.status}[/]"
            )
    return Panel(table, border_style="blue")


def get_market_status() -> str:
    now = datetime.now()
    h, m = now.hour, now.minute
    t = h * 60 + m
    if t < 9 * 60 + 15:         return "等待开盘"
    elif t < 9 * 60 + 25:       return "集合竞价"
    elif t < 11 * 60 + 30:      return "盘中交易"
    elif t < 13 * 60:            return "午休"
    elif t < 15 * 60:            return "盘中交易"
    else:                        return "已收盘"


def get_gpu_info() -> str:
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated(0) / 1024**2
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        return f"GPU: {mem_used:.0f}/{mem_total:.0f} MB | CUDA ✓"
    return "CPU 模式 (CUDA 未启用)"


# ============================================================
# 主扫描循环
# ============================================================
def run_dashboard(paper_mode=True, scan_interval=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = os.path.join(project_root, "models", "checkpoints", "best_echomarsh_model.pth")
    data_dir   = os.path.join(project_root, "data", "raw")

    console.print("[bold cyan]正在初始化 EchoMarsh 推理引擎...[/]")
    engine   = FastInferenceEngine(checkpoint, device)
    executor = QMTExecutor(paper_mode=paper_mode)

    console.print(f"[green]系统启动成功！设备: {device} | {'模拟模式' if paper_mode else '实盘模式'}[/]")
    console.print(f"[yellow]扫描间隔: {scan_interval}秒 | 按 Ctrl+C 退出[/]\n")

    signals   = []
    last_scan = 0

    layout = Layout()
    layout.split_column(
        Layout(name="header",    size=3),
        Layout(name="main",      ratio=1),
        Layout(name="bottom",    size=14),
    )
    layout["main"].split_row(
        Layout(name="signals",   ratio=3),
        Layout(name="positions", ratio=2),
    )

    with Live(layout, refresh_per_second=2, console=console) as live:
        while True:
            now = time.time()
            market_status = get_market_status()

            # 每隔 scan_interval 秒更新一次扫描信号
            if now - last_scan >= scan_interval:
                # 获取今日文件（实盘中应从 QMT 实时获取行情）
                today = datetime.now().strftime('%Y%m%d')
                files = glob.glob(os.path.join(data_dir, f"*_{today}.csv"))
                if not files:
                    # 非交易时间或无今日数据时，使用最新一批历史数据演示
                    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))[-20:]

                signals = engine.batch_score(files)
                last_scan = now

                # 根据当前时窗自动分发策略
                window_map = {
                    '等待开盘': None,
                    '集合竞价': '竞价',
                    '盘中交易': None,
                    '午休': None,
                    '已收盘': None
                }

                now_dt = datetime.now()
                h, m = now_dt.hour, now_dt.minute
                t = h * 60 + m
                if 9*60+15 <= t < 9*60+25:
                    trade_window = '竞价'
                elif 9*60+30 <= t < 10*60:
                    trade_window = '早盘'
                elif 10*60 <= t < 14*60+30:
                    trade_window = '日内'
                elif 14*60+30 <= t < 15*60:
                    trade_window = '尾盘'
                elif t >= 15*60 or t < 9*60+15:
                    trade_window = None
                else:
                    trade_window = None

                if trade_window:
                    executor.dispatch_strategy(signals, trade_window)

            gpu_info = get_gpu_info()

            # 更新各面板
            layout["header"].update(make_header(market_status))
            layout["signals"].update(make_signal_table(signals))
            layout["positions"].update(make_position_table(executor))
            layout["bottom"].update(make_order_table(executor))

            live.update(layout)
            time.sleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoMarsh Live Dashboard")
    parser.add_argument('--paper',    action='store_true', default=True, help='模拟模式（默认）')
    parser.add_argument('--live',     action='store_true', default=False, help='实盘模式')
    parser.add_argument('--interval', type=int, default=30, help='扫描间隔秒数')
    args = parser.parse_args()

    paper = not args.live
    try:
        run_dashboard(paper_mode=paper, scan_interval=args.interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]EchoMarsh 已安全退出。[/]")

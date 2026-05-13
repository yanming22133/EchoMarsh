"""
EchoMarsh 每日推荐报告 (Daily Report)
======================================
主入口脚本。收盘后运行，输出终端推荐表 + 导出精美 HTML 日报。

用法：
    python scripts/daily_report.py                   # 当日
    python scripts/daily_report.py --date 20260512   # 指定日期
"""

import os
import sys
import argparse
from datetime import datetime

# Windows 终端编码修复
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from core.scanner.daily_scanner import DailyScannerEngine
from core.scanner.persistence import PersistenceManager


console = Console(force_terminal=True)


def render_terminal_report(scored_stocks, market):
    """在终端渲染精美推荐表"""

    # 头部面板
    header = Text()
    header.append("⚡ EchoMarsh ", style="bold cyan")
    header.append("鸣泽量化", style="bold white")
    header.append(f"  │  {market.date[:4]}-{market.date[4:6]}-{market.date[6:]}  │  ", style="dim")
    header.append(f"大盘: {market.emoji} {market.label_cn}", style="bold green" if market.level == 'STRONG' else "bold yellow" if market.level == 'NEUTRAL' else "bold red")

    sub_info = Text()
    sub_info.append(f"  {market.summary_line()}", style="dim white")

    console.print(Panel(header, subtitle=sub_info, style="bold blue", box=box.DOUBLE))

    if not scored_stocks:
        console.print("[yellow]今日无符合条件的推荐标的。[/]")
        return

    # 推荐表
    table = Table(
        title=f"[bold yellow]📋 明日重点关注（主板，共 {len(scored_stocks)} 只）",
        box=box.ROUNDED, show_header=True, header_style="bold magenta",
        title_justify="left"
    )
    table.add_column("#", style="bold white", width=3, justify="center")
    table.add_column("代码", style="cyan", width=8)
    table.add_column("名称", style="bold white", width=10)
    table.add_column("行业", style="dim", width=10)
    table.add_column("连板", width=4, justify="center")
    table.add_column("热榜", width=8, justify="center")
    table.add_column("综合分", width=7, justify="center")
    table.add_column("信号", width=24)
    table.add_column("核心理由", width=36)

    for i, s in enumerate(scored_stocks):
        # 信号强度条
        bar_len = int(s.total_score / 100 * 16)
        bar = "█" * bar_len + "░" * (16 - bar_len)

        if s.total_score >= 80:
            level_label = "强烈推荐"
            score_style = "bold red"
            bar_style = "bold red"
        elif s.total_score >= 65:
            level_label = "推荐"
            score_style = "bold yellow"
            bar_style = "bold yellow"
        elif s.total_score >= 50:
            level_label = "关注"
            score_style = "green"
            bar_style = "green"
        else:
            level_label = "观望"
            score_style = "dim"
            bar_style = "dim"

        # 热榜标签
        if s.sentiment_tags:
            hot_label = Text(','.join(s.sentiment_tags[:2]), style="bold red")
        else:
            hot_label = Text("-", style="dim")

        # 连板样式
        lb_text = Text(f"{s.lianban}板", style="bold red" if s.lianban >= 3 else "yellow" if s.lianban == 2 else "white")

        # 核心理由（取得分最高的前2条）
        top_reasons = s.top_reasons(2)
        reason_text = ' | '.join([r.detail[:18] for r in top_reasons])

        table.add_row(
            str(i + 1),
            s.code,
            s.name,
            s.industry[:8],
            lb_text,
            hot_label,
            Text(f"{s.total_score:.0f}分", style=score_style),
            Text(f"{bar} {level_label}", style=bar_style),
            reason_text,
        )

    console.print(table)

    # 风险提示
    console.print()
    console.print("[dim]⚠  风险提示: 止损 -3% │ 目标 +5~8% │ 单票仓位 ≤ 30% │ 强势股追高需谨慎[/]")

    # 详细推荐理由
    console.print()
    console.print("[bold cyan]📝 详细推荐理由[/]")
    console.print("─" * 70)
    for i, s in enumerate(scored_stocks):
        console.print(f"\n[bold]#{i+1} {s.code} {s.name}[/] [dim]({s.industry})[/] — [bold]{s.total_score:.0f}分[/]")

        # 12 因子雷达
        factors = [
            ('连板', s.score_lianban, 15), ('封板', s.score_fengban, 10),
            ('资金', s.score_fund, 12), ('动量', s.score_momentum, 8),
            ('趋势', s.score_trend, 7), ('板块', s.score_sector, 10),
            ('情绪', s.score_sentiment, 10), ('龙虎', s.score_lhb, 8),
            ('换手', s.score_turnover, 5), ('背离', s.score_divergence, 5),
            ('市值', s.score_mcap, 5), ('安全', s.score_safety, 5),
        ]
        parts = []
        for fname, fscore, fmax in factors:
            ratio = fscore / fmax if fmax > 0 else 0
            if ratio >= 0.8:
                parts.append(f"[green]{fname}{fscore:.0f}[/]")
            elif ratio >= 0.5:
                parts.append(f"[yellow]{fname}{fscore:.0f}[/]")
            else:
                parts.append(f"[red]{fname}{fscore:.0f}[/]")
        console.print("  因子: " + " │ ".join(parts))

        # 推荐理由
        for r in s.top_reasons(4):
            console.print(f"  [green]✓[/] {r.detail}")

        # 风险警告
        warnings = s.risk_warnings()
        if warnings:
            for w in warnings:
                console.print(f"  [red]⚠[/] {w.detail}")


def generate_html_report(scored_stocks, market, output_path):
    """生成精美 HTML 日报"""

    stocks_rows = ""
    for i, s in enumerate(scored_stocks):
        bar_pct = int(s.total_score)
        if s.total_score >= 80:
            bar_color = "#ef4444"
            level = "强烈推荐"
        elif s.total_score >= 65:
            bar_color = "#f59e0b"
            level = "推荐"
        elif s.total_score >= 50:
            bar_color = "#22c55e"
            level = "关注"
        else:
            bar_color = "#94a3b8"
            level = "观望"

        hot_badge = ""
        for tag in s.sentiment_tags[:2]:
            hot_badge += f'<span class="badge hot">{tag}</span>'
        if not hot_badge:
            hot_badge = '<span class="badge neutral">-</span>'

        reasons_html = ""
        for r in s.top_reasons(4):
            reasons_html += f'<div class="reason good">✓ {r.detail}</div>'
        for w in s.risk_warnings():
            reasons_html += f'<div class="reason warn">⚠ {w.detail}</div>'

        # 因子雷达 mini bars
        factors = [
            ('连板', s.score_lianban, 15), ('封板', s.score_fengban, 10),
            ('资金', s.score_fund, 12), ('动量', s.score_momentum, 8),
            ('趋势', s.score_trend, 7), ('板块', s.score_sector, 10),
            ('情绪', s.score_sentiment, 10), ('龙虎', s.score_lhb, 8),
            ('换手', s.score_turnover, 5), ('背离', s.score_divergence, 5),
            ('市值', s.score_mcap, 5), ('安全', s.score_safety, 5),
        ]
        radar_html = '<div class="radar">'
        for fname, fscore, fmax in factors:
            pct = int(fscore / fmax * 100) if fmax > 0 else 0
            color = "#22c55e" if pct >= 80 else "#f59e0b" if pct >= 50 else "#ef4444"
            radar_html += f'<div class="radar-item"><span class="radar-label">{fname}</span><div class="radar-bar"><div class="radar-fill" style="width:{pct}%;background:{color}"></div></div></div>'
        radar_html += '</div>'

        stocks_rows += f"""
        <div class="stock-card">
            <div class="stock-header">
                <div class="stock-rank">#{i+1}</div>
                <div class="stock-info">
                    <div class="stock-name">{s.name}</div>
                    <div class="stock-code">{s.code} · {s.industry}</div>
                </div>
                <div class="stock-lianban">{s.lianban}板</div>
                <div class="stock-badges">{hot_badge}</div>
                <div class="stock-score" style="color:{bar_color}">
                    <div class="score-num">{s.total_score:.0f}</div>
                    <div class="score-label">{level}</div>
                </div>
            </div>
            <div class="score-bar-container">
                <div class="score-bar" style="width:{bar_pct}%;background:linear-gradient(90deg,{bar_color}88,{bar_color})"></div>
            </div>
            {radar_html}
            <div class="reasons">{reasons_html}</div>
        </div>"""

    market_color = "#22c55e" if market.level == 'STRONG' else "#f59e0b" if market.level == 'NEUTRAL' else "#ef4444"
    date_fmt = f"{market.date[:4]}-{market.date[4:6]}-{market.date[6:]}"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EchoMarsh 每日推荐 | {date_fmt}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI','Microsoft YaHei',sans-serif; background:#0f172a; color:#e2e8f0; min-height:100vh; padding:20px; }}
  .container {{ max-width:900px; margin:0 auto; }}
  .header {{ background:linear-gradient(135deg,#1e293b,#334155); border-radius:16px; padding:24px 32px; margin-bottom:20px; border:1px solid #334155; }}
  .header h1 {{ font-size:24px; margin-bottom:8px; }}
  .header h1 span {{ color:#06b6d4; }}
  .header .market-info {{ display:flex; gap:20px; flex-wrap:wrap; font-size:14px; color:#94a3b8; margin-top:12px; }}
  .header .market-level {{ font-size:18px; font-weight:bold; color:{market_color}; }}
  .stock-card {{ background:#1e293b; border-radius:12px; padding:20px; margin-bottom:16px; border:1px solid #334155; transition:transform 0.2s; }}
  .stock-card:hover {{ transform:translateY(-2px); border-color:#475569; }}
  .stock-header {{ display:flex; align-items:center; gap:16px; margin-bottom:12px; }}
  .stock-rank {{ font-size:20px; font-weight:bold; color:#64748b; min-width:36px; }}
  .stock-info {{ flex:1; }}
  .stock-name {{ font-size:18px; font-weight:bold; }}
  .stock-code {{ font-size:13px; color:#64748b; margin-top:2px; }}
  .stock-lianban {{ background:#dc2626; color:white; padding:4px 10px; border-radius:20px; font-size:13px; font-weight:bold; }}
  .stock-score {{ text-align:center; min-width:60px; }}
  .score-num {{ font-size:28px; font-weight:bold; }}
  .score-label {{ font-size:11px; margin-top:2px; }}
  .score-bar-container {{ background:#0f172a; border-radius:4px; height:6px; margin-bottom:14px; }}
  .score-bar {{ height:100%; border-radius:4px; transition:width 0.5s; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; margin-right:4px; }}
  .badge.hot {{ background:#dc262622; color:#f87171; border:1px solid #dc262644; }}
  .badge.neutral {{ background:#33415522; color:#64748b; border:1px solid #33415544; }}
  .radar {{ display:flex; flex-wrap:wrap; gap:6px 12px; margin-bottom:12px; }}
  .radar-item {{ display:flex; align-items:center; gap:4px; font-size:11px; min-width:100px; }}
  .radar-label {{ color:#64748b; min-width:24px; }}
  .radar-bar {{ flex:1; height:4px; background:#0f172a; border-radius:2px; min-width:40px; }}
  .radar-fill {{ height:100%; border-radius:2px; }}
  .reasons {{ }}
  .reason {{ font-size:13px; padding:4px 0; color:#94a3b8; }}
  .reason.good {{ color:#86efac; }}
  .reason.warn {{ color:#fca5a5; }}
  .footer {{ text-align:center; color:#475569; font-size:12px; margin-top:30px; padding:20px; }}
  .risk {{ background:#7f1d1d22; border:1px solid #7f1d1d44; border-radius:8px; padding:12px 16px; margin-top:20px; color:#fca5a5; font-size:13px; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1><span>⚡ EchoMarsh</span> 鸣泽量化 · 每日推荐</h1>
    <div class="market-info">
      <span>📅 {date_fmt}</span>
      <span class="market-level">{market.emoji} {market.label_cn}</span>
      <span>涨停 {market.zt_count} 只</span>
      <span>炸板率 {market.zb_ratio:.0%}</span>
      <span>跌停 {market.dt_count} 只</span>
      <span>连板王: {market.max_lianban_name}({market.max_lianban}板)</span>
    </div>
  </div>

  {stocks_rows}

  <div class="risk">
    ⚠ 风险提示: 止损 -3% │ 目标 +5~8% │ 单票仓位 ≤ 30% │ 本报告仅为技术分析参考，不构成投资建议
  </div>

  <div class="footer">
    EchoMarsh 鸣泽量化系统 · 12因子多维度评分 · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 生成
  </div>
</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n💾 HTML 日报已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="EchoMarsh 每日推荐报告")
    parser.add_argument('--date', default=None, help='指定日期 YYYYMMDD，默认今天')
    parser.add_argument('--board', default='main', choices=['main', 'all'],
                        help='main=仅主板60/00, all=全市场')
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime('%Y%m%d')

    console.print(f"[bold cyan]⚡ EchoMarsh 每日推荐报告 | {date_str}[/]")
    console.print()

    # 执行扫描
    engine = DailyScannerEngine(board_filter=args.board)
    scored_stocks, market = engine.scan(date_str)

    # 终端输出
    render_terminal_report(scored_stocks, market)

    # 存入 SQLite 持久化层
    pm = PersistenceManager()
    pm.save_recommendations(date_str, scored_stocks, market.label_cn)

    # HTML 日报
    output_dir = os.path.join(project_root, 'output')
    html_path = os.path.join(output_dir, f'report_{date_str}.html')
    generate_html_report(scored_stocks, market, html_path)


if __name__ == '__main__':
    main()

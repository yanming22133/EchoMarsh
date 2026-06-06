"""
EchoMarsh 因子评估模块 (Factor Evaluation)
==========================================
量化选股核心指标：IC / Rank IC / IR / 分层回测。

面试常问："你的因子 IC 多少？Rank IC 多少？"
这个模块直接回答这个问题。
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class FactorReport:
    """单因子/模型预测评估报告"""
    name: str = ""
    ic_mean: float = 0.0          # 均值 IC (Pearson)
    ic_std: float = 0.0           # IC 标准差
    ir: float = 0.0               # Information Ratio = IC_mean / IC_std
    ic_positive_ratio: float = 0.0  # IC > 0 的比例
    rank_ic_mean: float = 0.0     # 均值 Rank IC (Spearman)
    rank_ic_std: float = 0.0
    rank_ir: float = 0.0
    hit_rate: float = 0.0         # 方向预测正确率
    long_short_ret: float = 0.0   # 多空收益差 (top - bottom decile)
    top_decile_ret: float = 0.0   # Top 10% 平均收益
    bottom_decile_ret: float = 0.0  # Bottom 10% 平均收益
    ic_series: List[float] = field(default_factory=list)
    n_periods: int = 0

    def summary(self) -> str:
        lines = [
            f"========== {self.name} 因子评估 ==========",
            f"样本期数: {self.n_periods}",
            f"--- IC (Pearson) ---",
            f"  IC Mean:  {self.ic_mean:.4f}",
            f"  IC Std:   {self.ic_std:.4f}",
            f"  IR:       {self.ir:.4f}  {'✅ 优秀' if self.ir > 0.5 else '⚠️ 一般' if self.ir > 0.2 else '❌ 较弱'}",
            f"  IC>0 占比: {self.ic_positive_ratio:.1%}",
            f"--- Rank IC (Spearman) ---",
            f"  Rank IC Mean: {self.rank_ic_mean:.4f}",
            f"  Rank IC Std:  {self.rank_ic_std:.4f}",
            f"  Rank IR:      {self.rank_ir:.4f}",
            f"--- 方向准确率 ---",
            f"  Hit Rate: {self.hit_rate:.1%}",
            f"--- 分层收益 ---",
            f"  Top 10%:     {self.top_decile_ret:+.4f}",
            f"  Bottom 10%:  {self.bottom_decile_ret:+.4f}",
            f"  Long-Short:  {self.long_short_ret:+.4f}",
            f"=" * 44,
        ]
        return "\n".join(lines)


def compute_ic(predictions: np.ndarray, actual_returns: np.ndarray) -> Tuple[float, float]:
    """
    计算 IC (Information Coefficient) 和 Rank IC。

    Args:
        predictions: 模型预测值 (N,) 或 (N, T)
        actual_returns: 实际收益 (N,)

    Returns:
        (ic_pearson, ic_spearman)
    """
    if len(predictions) < 3:
        return 0.0, 0.0

    # Pearson IC
    ic_pearson, _ = stats.pearsonr(predictions, actual_returns)
    # Spearman Rank IC
    ic_spearman, _ = stats.spearmanr(predictions, actual_returns)

    return float(ic_pearson), float(ic_spearman)


def compute_cross_sectional_ic(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    stock_ids: Optional[np.ndarray] = None,
    dates: Optional[np.ndarray] = None,
) -> FactorReport:
    """
    横截面 IC 分析 — 按日期分组，每期计算一次 IC。

    这是量化选股的标准评估方式：
    每个交易日，用预测值排序选股，计算预测与实际收益的相关性。

    Args:
        predictions: 模型预测 (N,)
        actual_returns: 实际收益 (N,)
        stock_ids: 股票代码 (N,)，可选
        dates: 日期 (N,)，若为 None 则视为单一横截面

    Returns:
        FactorReport 包含全部统计指标
    """
    if dates is None:
        # 单一横截面
        ic, rank_ic = compute_ic(predictions, actual_returns)
        hit = float(np.mean(np.sign(predictions) == np.sign(actual_returns)))

        # 分层收益
        n = len(predictions)
        if n >= 10:
            order = np.argsort(predictions)
            top_n = max(1, n // 10)
            top_ret = float(np.mean(actual_returns[order[-top_n:]]))
            bot_ret = float(np.mean(actual_returns[order[:top_n]]))
        else:
            top_ret, bot_ret = 0.0, 0.0

        return FactorReport(
            name="single_period",
            ic_mean=ic, ic_std=0.0, ir=float('inf') if ic > 0 else 0.0,
            ic_positive_ratio=1.0 if ic > 0 else 0.0,
            rank_ic_mean=rank_ic, rank_ic_std=0.0, rank_ir=float('inf') if rank_ic > 0 else 0.0,
            hit_rate=hit,
            long_short_ret=top_ret - bot_ret,
            top_decile_ret=top_ret, bottom_decile_ret=bot_ret,
            ic_series=[ic], n_periods=1,
        )

    # 多期横截面
    unique_dates = np.unique(dates)
    ic_list, rank_ic_list, hit_list = [], [], []
    top_rets, bot_rets = [], []

    for d in unique_dates:
        mask = dates == d
        pred_d = predictions[mask]
        ret_d = actual_returns[mask]

        if len(pred_d) < 10:
            continue

        ic, rank_ic = compute_ic(pred_d, ret_d)
        ic_list.append(ic)
        rank_ic_list.append(rank_ic)

        hit = np.mean(np.sign(pred_d) == np.sign(ret_d))
        hit_list.append(hit)

        # 分层
        order = np.argsort(pred_d)
        top_n = max(1, len(pred_d) // 10)
        top_rets.append(np.mean(ret_d[order[-top_n:]]))
        bot_rets.append(np.mean(ret_d[order[:top_n]]))

    ic_arr = np.array(ic_list)
    rank_ic_arr = np.array(rank_ic_list)

    return FactorReport(
        name="cross_sectional",
        ic_mean=float(np.mean(ic_arr)),
        ic_std=float(np.std(ic_arr, ddof=1)),
        ir=float(np.mean(ic_arr) / (np.std(ic_arr, ddof=1) + 1e-10)),
        ic_positive_ratio=float(np.mean(ic_arr > 0)),
        rank_ic_mean=float(np.mean(rank_ic_arr)),
        rank_ic_std=float(np.std(rank_ic_arr, ddof=1)),
        rank_ir=float(np.mean(rank_ic_arr) / (np.std(rank_ic_arr, ddof=1) + 1e-10)),
        hit_rate=float(np.mean(hit_list)),
        long_short_ret=float(np.mean(top_rets) - np.mean(bot_rets)),
        top_decile_ret=float(np.mean(top_rets)),
        bottom_decile_ret=float(np.mean(bot_rets)),
        ic_series=[float(x) for x in ic_arr],
        n_periods=len(ic_arr),
    )


def compute_decile_returns(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    分层回测 — 按预测值从低到高分组，计算每组的平均收益。

    理想结果：收益随分位单调递增（预测越高中收益越高）。

    Returns:
        DataFrame with columns: decile, mean_ret, count
    """
    n = len(predictions)
    if n < n_deciles:
        return pd.DataFrame()

    order = np.argsort(predictions)
    decile_size = n // n_deciles

    rows = []
    for d in range(n_deciles):
        start = d * decile_size
        end = start + decile_size if d < n_deciles - 1 else n
        idx = order[start:end]
        rows.append({
            'decile': d + 1,
            'mean_ret': float(np.mean(actual_returns[idx])),
            'count': len(idx),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 演示：用随机数据跑一遍评估流程
    print("=== 因子评估模块自测 ===\n")

    np.random.seed(42)
    N = 1000
    # 模拟：预测值有一定预测能力的场景 (IC ~ 0.05)
    true_signal = np.random.randn(N)
    predictions = true_signal + np.random.randn(N) * 3.0
    actual_returns = true_signal * 0.15 + np.random.randn(N) * 4.0

    report = compute_cross_sectional_ic(predictions, actual_returns)
    print(report.summary())

    print("\n--- 分层回测 ---")
    decile_df = compute_decile_returns(predictions, actual_returns)
    print(decile_df.to_string(index=False))

"""
EchoMarsh 大盘情绪模块 (Market Sentiment Engine)
=================================================
通过全市场统计数据判断当日大盘情绪等级：
  STRONG  — 强势（涨停多、炸板少、跌停少）→ 可激进操作
  NEUTRAL — 震荡（分化明显）→ 精选个股
  WEAK    — 弱势（跌停多、炸板高、涨停少）→ 收窄推荐 / 空仓观望

数据来源（全免费 akshare）：
  - stock_zt_pool_em()      涨停板池
  - stock_zt_pool_zbgc_em() 炸板（曾涨停后开板）
  - stock_zt_pool_dtgc_em() 跌停板池
"""

import os
import traceback
from datetime import datetime
from dataclasses import dataclass

os.environ["NO_PROXY"] = "*"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

import akshare as ak


@dataclass
class MarketSentimentResult:
    """大盘情绪评估结果"""
    date: str
    level: str              # 'STRONG', 'NEUTRAL', 'WEAK'
    zt_count: int           # 涨停数量
    zb_count: int           # 炸板数量
    dt_count: int           # 跌停数量
    zb_ratio: float         # 炸板率 = 炸板/(涨停+炸板)
    max_lianban: int        # 最高连板数
    max_lianban_name: str   # 最高连板股名称
    max_lianban_code: str   # 最高连板股代码
    recommend_slots: int    # 建议推荐数量

    @property
    def emoji(self):
        return {'STRONG': '🔥', 'NEUTRAL': '⚖️', 'WEAK': '❄️'}.get(self.level, '❓')

    @property
    def label_cn(self):
        return {'STRONG': '强势', 'NEUTRAL': '震荡', 'WEAK': '弱势'}.get(self.level, '未知')

    def summary_line(self):
        return (f"涨停 {self.zt_count} 只 | 炸板率 {self.zb_ratio:.0%} | "
                f"跌停 {self.dt_count} 只 | 连板王: {self.max_lianban_name}({self.max_lianban}板)")


class MarketSentimentEngine:
    """
    大盘情绪判断引擎。
    调用方式：result = MarketSentimentEngine().analyze(date_str)
    """

    def analyze(self, date_str: str = None) -> MarketSentimentResult:
        """
        分析指定日期的大盘情绪。
        :param date_str: 格式 'YYYYMMDD'，默认今天
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y%m%d')

        print(f"[大盘情绪] 正在分析 {date_str} 的市场状态...")

        zt_count = 0
        zb_count = 0
        dt_count = 0
        max_lianban = 0
        max_lianban_name = '-'
        max_lianban_code = '-'

        # 1. 涨停板池
        try:
            zt_df = ak.stock_zt_pool_em(date=date_str)
            if zt_df is not None and not zt_df.empty:
                zt_count = len(zt_df)
                if '连板数' in zt_df.columns:
                    idx_max = zt_df['连板数'].idxmax()
                    max_lianban = int(zt_df.loc[idx_max, '连板数'])
                    max_lianban_name = str(zt_df.loc[idx_max, '名称']) if '名称' in zt_df.columns else '-'
                    max_lianban_code = str(zt_df.loc[idx_max, '代码']) if '代码' in zt_df.columns else '-'
            print(f"  涨停: {zt_count} 只, 最高连板: {max_lianban}板 ({max_lianban_name})")
        except Exception as e:
            print(f"  [涨停池] 获取失败: {e}")

        # 2. 炸板池
        try:
            zb_df = ak.stock_zt_pool_zbgc_em(date=date_str)
            if zb_df is not None and not zb_df.empty:
                zb_count = len(zb_df)
            print(f"  炸板: {zb_count} 只")
        except Exception as e:
            print(f"  [炸板池] 获取失败: {e}")

        # 3. 跌停池
        try:
            dt_df = ak.stock_zt_pool_dtgc_em(date=date_str)
            if dt_df is not None and not dt_df.empty:
                dt_count = len(dt_df)
            print(f"  跌停: {dt_count} 只")
        except Exception as e:
            print(f"  [跌停池] 获取失败: {e}")

        # 计算炸板率
        total_attempt = zt_count + zb_count
        zb_ratio = zb_count / total_attempt if total_attempt > 0 else 0.0

        # 判断情绪等级
        level, slots = self._judge_level(zt_count, zb_ratio, dt_count)

        result = MarketSentimentResult(
            date=date_str,
            level=level,
            zt_count=zt_count,
            zb_count=zb_count,
            dt_count=dt_count,
            zb_ratio=zb_ratio,
            max_lianban=max_lianban,
            max_lianban_name=max_lianban_name,
            max_lianban_code=max_lianban_code,
            recommend_slots=slots,
        )
        print(f"[大盘情绪] 结论: {result.emoji} {result.label_cn} | {result.summary_line()}")
        return result

    def _judge_level(self, zt_count, zb_ratio, dt_count):
        """
        判断情绪等级，返回 (level, recommend_slots)。

        强势条件（满足任意两项）：
          - 涨停 >= 60 只
          - 炸板率 < 15%
          - 跌停 < 5 只

        弱势条件（满足任意两项）：
          - 涨停 < 30 只
          - 炸板率 > 30%
          - 跌停 > 15 只
        """
        strong_signals = 0
        weak_signals = 0

        if zt_count >= 60:
            strong_signals += 1
        if zt_count < 30:
            weak_signals += 1

        if zb_ratio < 0.15:
            strong_signals += 1
        if zb_ratio > 0.30:
            weak_signals += 1

        if dt_count < 5:
            strong_signals += 1
        if dt_count > 15:
            weak_signals += 1

        if strong_signals >= 2:
            return 'STRONG', 8    # 强势市场：推荐上限 8 只
        elif weak_signals >= 2:
            return 'WEAK', 3      # 弱势市场：最多推荐 3 只，降低暴露
        else:
            return 'NEUTRAL', 5   # 震荡市场：精选 5 只


if __name__ == "__main__":
    engine = MarketSentimentEngine()
    result = engine.analyze()
    print(f"\n完整结果: {result}")

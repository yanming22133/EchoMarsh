"""
EchoMarsh 全球市场与新闻模块 (Global Market & News)
====================================================
拉取影响 A 股决策的外围信息：
  1. 美股三大指数（道琼斯、纳斯达克、标普500）
  2. 富时中国A50期货（最直接的A股风向标）
  3. 港股恒生指数
  4. A股重要新闻快讯
  5. 重要经济数据/政策信号

数据源：全部通过 akshare 免费获取
"""

import os
import traceback
from datetime import datetime
from dataclasses import dataclass, field
from typing import List

os.environ["NO_PROXY"] = "*"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

import akshare as ak
import pandas as pd


@dataclass
class IndexSnapshot:
    """指数快照"""
    name: str
    code: str
    price: float = 0.0
    change_pct: float = 0.0   # 涨跌幅%
    status: str = ''           # 上涨/下跌/平盘

    @property
    def emoji(self):
        if self.change_pct > 0.5:
            return '🟢'
        elif self.change_pct < -0.5:
            return '🔴'
        return '⚪'

    def one_liner(self):
        return f"{self.emoji} {self.name}: {self.price:.2f} ({self.change_pct:+.2f}%)"


@dataclass
class NewsItem:
    """新闻条目"""
    time: str
    title: str
    source: str = ''
    importance: str = 'normal'  # normal / important / critical


@dataclass
class GlobalMarketReport:
    """全球市场综合报告"""
    us_indices: List[IndexSnapshot] = field(default_factory=list)
    hk_indices: List[IndexSnapshot] = field(default_factory=list)
    a50_future: IndexSnapshot = None
    news_items: List[NewsItem] = field(default_factory=list)
    overall_signal: str = ''    # POSITIVE / NEUTRAL / NEGATIVE
    signal_reason: str = ''

    def summary(self):
        parts = []
        for idx in self.us_indices:
            parts.append(idx.one_liner())
        if self.a50_future:
            parts.append(self.a50_future.one_liner())
        for idx in self.hk_indices:
            parts.append(idx.one_liner())
        return '\n'.join(parts)


class GlobalMarketEngine:
    """全球市场信息引擎"""

    def analyze(self) -> GlobalMarketReport:
        """综合拉取全球市场信息"""
        report = GlobalMarketReport()

        print("[全球市场] 拉取外围市场数据...")
        report.us_indices = self._get_us_indices()
        report.hk_indices = self._get_hk_indices()
        report.a50_future = self._get_a50_future()
        report.news_items = self._get_news()

        # 综合判断外围信号
        report.overall_signal, report.signal_reason = self._judge_signal(report)

        return report

    def _get_us_indices(self) -> List[IndexSnapshot]:
        """美股三大指数"""
        indices = []
        mapping = {
            '.DJI': '道琼斯',
            '.IXIC': '纳斯达克',
            '.INX': '标普500',
        }
        try:
            for code, name in mapping.items():
                try:
                    df = ak.index_us_stock_sina(symbol=code)
                    if df is not None and not df.empty:
                        last = df.iloc[-1]
                        close_col = None
                        for c in ['close', '收盘', 'last']:
                            if c in df.columns:
                                close_col = c
                                break
                        if close_col is None and len(df.columns) >= 2:
                            close_col = df.columns[1]

                        price = float(last[close_col]) if close_col else 0
                        # 计算涨跌幅
                        if len(df) >= 2:
                            prev = float(df.iloc[-2][close_col]) if close_col else 0
                            pct = (price - prev) / prev * 100 if prev > 0 else 0
                        else:
                            pct = 0
                        indices.append(IndexSnapshot(name=name, code=code, price=price, change_pct=pct))
                except Exception as e:
                    indices.append(IndexSnapshot(name=name, code=code, status=f'获取失败: {e}'))
        except Exception as e:
            print(f"  [美股] 获取失败: {e}")
        return indices

    def _get_hk_indices(self) -> List[IndexSnapshot]:
        """港股恒生指数"""
        indices = []
        try:
            df = ak.stock_hk_index_daily_em(symbol="HSI")
            if df is not None and not df.empty:
                last = df.iloc[-1]
                price = float(last.get('收盘', last.get('close', 0)))
                if len(df) >= 2:
                    prev = float(df.iloc[-2].get('收盘', df.iloc[-2].get('close', 0)))
                    pct = (price - prev) / prev * 100 if prev > 0 else 0
                else:
                    pct = 0
                indices.append(IndexSnapshot(name='恒生指数', code='HSI', price=price, change_pct=pct))
        except Exception as e:
            print(f"  [港股] 获取失败: {e}")
            indices.append(IndexSnapshot(name='恒生指数', code='HSI', status='获取失败'))
        return indices

    def _get_a50_future(self) -> IndexSnapshot:
        """富时中国A50期货（最直接的A股隔夜风向标）"""
        try:
            df = ak.futures_foreign_hist(symbol="A50")
            if df is not None and not df.empty:
                last = df.iloc[-1]
                close_col = None
                for c in ['收盘', 'close', '收盘价']:
                    if c in df.columns:
                        close_col = c
                        break
                if close_col is None and len(df.columns) >= 5:
                    close_col = df.columns[4]

                price = float(last[close_col]) if close_col else 0
                if len(df) >= 2:
                    prev = float(df.iloc[-2][close_col]) if close_col else 0
                    pct = (price - prev) / prev * 100 if prev > 0 else 0
                else:
                    pct = 0
                return IndexSnapshot(name='A50期货', code='A50', price=price, change_pct=pct)
        except Exception as e:
            print(f"  [A50] 获取失败: {e}")
        return IndexSnapshot(name='A50期货', code='A50', status='获取失败')

    def _get_news(self) -> List[NewsItem]:
        """获取 A 股重要新闻"""
        news = []
        try:
            # 东方财富快讯
            df = ak.stock_news_em(symbol="300059")  # 用一个通用代码触发快讯列表
            if df is not None and not df.empty:
                for _, row in df.head(10).iterrows():
                    title = str(row.get('新闻标题', row.get('title', '')))
                    time_str = str(row.get('发布时间', row.get('datetime', '')))
                    if title:
                        importance = 'normal'
                        for kw in ['央行', '降准', '降息', '国务院', '证监会', '美联储',
                                   '加关税', '制裁', '暴跌', '熔断']:
                            if kw in title:
                                importance = 'critical'
                                break
                        for kw in ['利好', '突破', '新高', '增持', '回购']:
                            if kw in title:
                                importance = 'important'
                                break
                        news.append(NewsItem(time=time_str, title=title, importance=importance))
        except Exception as e:
            print(f"  [新闻] 获取失败: {e}")

        # 备选：财联社电报
        try:
            df2 = ak.stock_zh_a_alerts_cls()
            if df2 is not None and not df2.empty:
                title_col = None
                for c in ['标题', 'title', '内容']:
                    if c in df2.columns:
                        title_col = c
                        break
                time_col = None
                for c in ['时间', 'datetime', 'time']:
                    if c in df2.columns:
                        time_col = c
                        break
                if title_col:
                    for _, row in df2.head(15).iterrows():
                        title = str(row[title_col])[:80]
                        time_str = str(row[time_col]) if time_col else ''
                        if title and len(title) > 5:
                            news.append(NewsItem(time=time_str, title=title, source='财联社'))
        except Exception:
            pass

        return news[:20]  # 最多返回20条

    def _judge_signal(self, report: GlobalMarketReport):
        """根据外围市场综合判断对 A 股的影响"""
        positive = 0
        negative = 0
        reasons = []

        # 美股
        for idx in report.us_indices:
            if idx.change_pct > 0.5:
                positive += 1
                reasons.append(f"{idx.name}涨{idx.change_pct:.1f}%")
            elif idx.change_pct < -0.5:
                negative += 1
                reasons.append(f"{idx.name}跌{abs(idx.change_pct):.1f}%")

        # A50
        if report.a50_future and report.a50_future.change_pct != 0:
            if report.a50_future.change_pct > 0.3:
                positive += 2  # A50 权重更高
                reasons.append(f"A50期货涨{report.a50_future.change_pct:.1f}%")
            elif report.a50_future.change_pct < -0.3:
                negative += 2
                reasons.append(f"A50期货跌{abs(report.a50_future.change_pct):.1f}%")

        # 港股
        for idx in report.hk_indices:
            if idx.change_pct > 0.5:
                positive += 1
            elif idx.change_pct < -0.5:
                negative += 1

        if positive >= 3:
            return 'POSITIVE', '外围普涨，' + '，'.join(reasons[:3])
        elif negative >= 3:
            return 'NEGATIVE', '外围普跌，' + '，'.join(reasons[:3])
        else:
            return 'NEUTRAL', '外围分化，' + '，'.join(reasons[:3]) if reasons else '外围平静'


class PortfolioAdvisor:
    """
    持仓顾问 - 根据当前持仓 + 最新行情，给出操作建议。
    """

    def advise_position(self, code: str, name: str, cost_price: float,
                        volume: int, buy_date: str) -> dict:
        """
        对单只持仓给出操作建议。

        返回: {
            'action': 'HOLD' / 'SELL' / 'ADD' / 'T_HIGH_SELL',
            'reason': str,
            'current_price': float,
            'pnl_pct': float,
            'details': [str]
        }
        """
        code = str(code).zfill(6)
        result = {
            'code': code, 'name': name, 'action': 'HOLD',
            'reason': '', 'current_price': 0, 'pnl_pct': 0, 'details': []
        }

        try:
            # 获取最新行情
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=30)).strftime('%Y%m%d')
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                     start_date=start_date, end_date=end_date, adjust="qfq")

            if df is None or df.empty:
                result['reason'] = '无法获取行情数据'
                return result

            cur_price = float(df['收盘'].iloc[-1])
            result['current_price'] = cur_price
            pnl_pct = (cur_price - cost_price) / cost_price * 100
            result['pnl_pct'] = pnl_pct

            close = df['收盘'].values
            vol = df['成交量'].values

            # 均线
            ma5 = float(pd.Series(close).rolling(5).mean().iloc[-1]) if len(close) >= 5 else cur_price
            ma10 = float(pd.Series(close).rolling(10).mean().iloc[-1]) if len(close) >= 10 else cur_price
            ma20 = float(pd.Series(close).rolling(20).mean().iloc[-1]) if len(close) >= 20 else cur_price

            details = []

            # ===== 止损规则 =====
            if pnl_pct <= -5:
                result['action'] = 'SELL'
                result['reason'] = f'触发止损线！浮亏 {pnl_pct:.1f}%，建议果断止损'
                details.append(f'成本 {cost_price:.2f} → 现价 {cur_price:.2f}，亏损 {pnl_pct:.1f}%')
                details.append('纪律第一：严格执行 -5% 止损，保护本金')
                result['details'] = details
                return result

            if pnl_pct <= -3:
                details.append(f'⚠️ 浮亏 {pnl_pct:.1f}%，接近止损线(-5%)，密切关注')

            # ===== 止盈/做T规则 =====
            if pnl_pct >= 8:
                result['action'] = 'SELL'
                result['reason'] = f'浮盈 {pnl_pct:.1f}%，达到目标价，建议止盈'
                details.append('落袋为安，利润达标不贪心')

            elif pnl_pct >= 5:
                # 看趋势决定是继续持有还是部分止盈
                if cur_price > ma5 > ma10:
                    result['action'] = 'HOLD'
                    result['reason'] = f'浮盈 {pnl_pct:.1f}%，均线多头继续持有'
                    details.append(f'股价在5日线({ma5:.2f})上方，趋势良好')
                    details.append('可设移动止盈：跌破5日线则减仓')
                else:
                    result['action'] = 'T_HIGH_SELL'
                    result['reason'] = f'浮盈 {pnl_pct:.1f}%，高抛 1/3 锁定利润'
                    details.append('均线走弱，建议做T高抛部分仓位')

            elif pnl_pct >= 3:
                result['action'] = 'HOLD'
                result['reason'] = f'浮盈 {pnl_pct:.1f}%，持有观察'
                details.append('设定止盈目标 +8%，止损移至成本价')

            elif pnl_pct >= 0:
                if cur_price > ma5:
                    result['action'] = 'HOLD'
                    result['reason'] = '微盈，站上5日线，继续观察'
                else:
                    result['action'] = 'HOLD'
                    result['reason'] = '微盈但跌破5日线，需密切关注'
                    details.append('若明日继续走弱，考虑减仓')

            else:
                # 小幅浮亏
                if cur_price > ma5 and cur_price > ma10:
                    result['action'] = 'HOLD'
                    result['reason'] = f'小幅浮亏 {pnl_pct:.1f}%，但均线支撑尚在'
                else:
                    result['action'] = 'HOLD'
                    result['reason'] = f'浮亏 {pnl_pct:.1f}%，均线走弱，考虑减仓'
                    details.append('跌破均线系统，反弹阻力增大')

            # 量价分析
            if len(vol) >= 6:
                vol_ratio = vol[-1] / (sum(vol[-6:-1]) / 5 + 1e-8)
                if vol_ratio > 2.5 and pnl_pct > 3:
                    details.append(f'⚠️ 量比 {vol_ratio:.1f}，放量异常，警惕主力出货')
                elif vol_ratio < 0.5:
                    details.append(f'缩量（量比 {vol_ratio:.1f}），观望为主')

            result['details'] = details

        except Exception as e:
            result['reason'] = f'分析失败: {e}'

        return result

    def advise_all_positions(self, positions: list) -> list:
        """对所有持仓逐一给出建议"""
        advices = []
        for pos in positions:
            advice = self.advise_position(
                code=pos.code, name=pos.name,
                cost_price=pos.cost_price, volume=pos.volume,
                buy_date=pos.buy_date
            )
            advices.append(advice)
        return advices


if __name__ == "__main__":
    # 测试全球市场
    gm = GlobalMarketEngine()
    report = gm.analyze()
    print("\n=== 全球市场 ===")
    print(report.summary())
    print(f"综合信号: {report.overall_signal} ({report.signal_reason})")

    print(f"\n=== 新闻 ({len(report.news_items)} 条) ===")
    for n in report.news_items[:5]:
        imp_mark = '🔴' if n.importance == 'critical' else '🟡' if n.importance == 'important' else '  '
        print(f"  {imp_mark} [{n.time}] {n.title}")

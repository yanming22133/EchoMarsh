"""
EchoMarsh 多因子打分引擎 (Multi-Factor Scoring Engine)
======================================================
核心模块：整合全部因子维度，对候选股进行综合评分并生成推荐理由。

12 大因子组（满分 100 分）：
  A. 连板强度        (15分) — 涨停板池连板数
  B. 封板质量        (10分) — 封单金额/流通市值、炸板回封次数
  C. 主力资金        (12分) — 主力净流入占比
  D. 技术形态-动量   (8分)  — RSI/MACD/量比
  E. 技术形态-趋势   (7分)  — 均线多头排列/突破
  F. 板块共振        (10分) — 行业涨幅排名
  G. 社区情绪        (10分) — 东财热度/飙升
  H. 龙虎榜席位      (8分)  — 拉萨天团扣分/顶级游资加分
  I. 换手率异动      (5分)  — 放量滞涨检测(反向扣分)
  J. 情绪资金背离    (5分)  — 散户狂热+主力出逃=杀猪盘(反向扣分)
  K. 市值适配        (5分)  — 20~200亿流通市值最优
  L. 安全过滤        (5分)  — ST/退市/次新股过滤
"""

import os
import time
import traceback
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional

os.environ["NO_PROXY"] = "*"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

import akshare as ak
import pandas as pd
import numpy as np

# curl_cffi for East Money subdomains that block Python requests TLS fingerprint
try:
    from curl_cffi import requests as curl_requests
    _curl_available = True
except ImportError:
    curl_requests = None
    _curl_available = False

# Patch akshare's request_with_retry to use curl_cffi (fixes 17.push2.eastmoney.com)
if _curl_available:
    import akshare.utils.request as _ak_req

    def _patched_request_with_retry(
        url, params=None, timeout=15, max_retries=3, base_delay=1.0,
        random_delay_range=(0.5, 1.5),
    ):
        last_exception = None
        for attempt in range(max_retries):
            try:
                with curl_requests.Session() as session:
                    response = session.get(url, params=params, timeout=timeout,
                                           impersonate="chrome131")
                    response.raise_for_status()
                    return response
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    import random
                    delay = base_delay * (2 ** attempt) + random.uniform(*random_delay_range)
                    time.sleep(delay)
        raise last_exception

    _ak_req.request_with_retry = _patched_request_with_retry
    # func.py imports request_with_retry by reference — patch there too
    import akshare.utils.func as _ak_func
    _ak_func.request_with_retry = _patched_request_with_retry

from core.scanner.community_sentiment import CommunitySentimentEngine
from core.scanner.market_sentiment import MarketSentimentEngine


@dataclass
class RecommendReason:
    """单条推荐理由"""
    factor: str       # 因子名
    score: float      # 该因子得分
    max_score: float  # 该因子满分
    detail: str       # 人话解释

    def __repr__(self):
        return f"  [{self.factor}] {self.score:.1f}/{self.max_score:.0f} — {self.detail}"


@dataclass
class ScoredStock:
    """评分完成的候选股"""
    code: str
    name: str
    industry: str
    lianban: int
    total_score: float
    reasons: List[RecommendReason] = field(default_factory=list)
    # 详细分项
    score_lianban: float = 0.0
    score_fengban: float = 0.0
    score_fund: float = 0.0
    score_momentum: float = 0.0
    score_trend: float = 0.0
    score_sector: float = 0.0
    score_sentiment: float = 0.0
    score_lhb: float = 0.0
    score_turnover: float = 0.0
    score_divergence: float = 0.0
    score_mcap: float = 0.0
    score_safety: float = 0.0
    # 附加信息
    sentiment_tags: list = field(default_factory=list)
    close_price: float = 0.0
    pct_change: float = 0.0
    turnover_rate: float = 0.0
    circulating_mcap: float = 0.0  # 亿元

    def top_reasons(self, n=5):
        """返回得分最高的前 N 条推荐理由"""
        sorted_r = sorted(self.reasons, key=lambda r: r.score, reverse=True)
        return sorted_r[:n]

    def risk_warnings(self):
        """返回所有扣分项作为风险提示"""
        return [r for r in self.reasons if r.score < 0]


class DailyScannerEngine:
    """
    每日多因子扫描引擎。
    用法：
        engine = DailyScannerEngine()
        results = engine.scan(date_str='20260513')
    """

    def __init__(self, board_filter='main'):
        """
        :param board_filter: 'main' = 只看60/00主板, 'all' = 全市场含创业板/科创板/北交所
        """
        self.board_filter = board_filter
        self.sentiment_engine = CommunitySentimentEngine()
        self.market_engine = MarketSentimentEngine()

    def scan(self, date_str: str = None) -> tuple:
        """
        执行全量扫描，返回 (scored_stocks, market_result)
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y%m%d')

        print("=" * 60)
        print(f"  EchoMarsh 多因子扫描引擎 | {date_str}")
        print("=" * 60)

        # 1. 大盘情绪
        market = self.market_engine.analyze(date_str)

        # 2. 社区情绪（批量预加载）
        self.sentiment_engine.load()

        # 3. 获取涨停板池作为候选股池
        print("\n[扫描] 获取涨停板池...")
        candidates = self._get_candidates(date_str)
        if not candidates:
            print("[扫描] 今日无候选股票")
            return [], market

        print(f"[扫描] 候选股 {len(candidates)} 只，开始逐票打分...")

        # 4. 获取行业板块排名（一次性拉取）
        sector_ranks = self._get_sector_ranks(date_str)

        # 5. 逐票打分
        scored = []
        for i, cand in enumerate(candidates):
            code = cand['code']
            name = cand['name']
            print(f"  ({i+1}/{len(candidates)}) 评分: {code} {name}...", end='')

            try:
                stock = self._score_stock(cand, sector_ranks, date_str)
                scored.append(stock)
                print(f" → {stock.total_score:.1f}分")
            except Exception as e:
                print(f" → 失败: {e}")

            # 防限速
            if (i + 1) % 5 == 0:
                time.sleep(1)

        # 6. 按总分排序
        scored.sort(key=lambda s: s.total_score, reverse=True)

        # 7. 根据大盘情绪裁剪推荐数量
        max_slots = market.recommend_slots
        final = scored[:max_slots]

        print(f"\n[扫描完成] 总评分 {len(scored)} 只 → 推荐 {len(final)} 只 (大盘: {market.label_cn})")
        return final, market

    def _get_candidates(self, date_str):
        """从涨停板池获取候选股"""
        candidates = []
        try:
            zt_df = ak.stock_zt_pool_em(date=date_str)
            if zt_df is None or zt_df.empty:
                return []

            for _, row in zt_df.iterrows():
                code = str(row.get('代码', '')).zfill(6)
                name = str(row.get('名称', ''))

                # 板块过滤
                if self.board_filter == 'main':
                    if not (code.startswith('60') or code.startswith('00')):
                        continue

                # ST 过滤
                if 'ST' in name.upper():
                    continue

                candidates.append({
                    'code': code,
                    'name': name,
                    'lianban': int(row.get('连板数', 1)) if '连板数' in zt_df.columns else 1,
                    'pct_change': float(row.get('涨跌幅', 0)) if '涨跌幅' in zt_df.columns else 0,
                    'turnover_rate': float(row.get('换手率', 0)) if '换手率' in zt_df.columns else 0,
                    'close': float(row.get('最新价', 0)) if '最新价' in zt_df.columns else 0,
                    'fengdan': float(row.get('封单资金', 0)) if '封单资金' in zt_df.columns else 0,
                    'liutong_mv': float(row.get('流通市值', 0)) if '流通市值' in zt_df.columns else 0,
                    'industry': str(row.get('所属行业', '未知')) if '所属行业' in zt_df.columns else '未知',
                })
        except Exception as e:
            print(f"[候选股] 获取失败: {e}")
            traceback.print_exc()

        return candidates

    def _fetch_fund_flow(self, code: str):
        """使用 curl_cffi 获取个股资金流向 (替代 ak.stock_individual_fund_flow)"""
        market_id = 1 if code.startswith('6') else 0
        url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
        params = {
            "lmt": "0",
            "klt": "101",
            "secid": f"{market_id}.{code}",
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
            "ut": "b2884a393a59ad64002292a3e90d46a5",
            "_": int(time.time() * 1000),
        }
        r = curl_requests.get(url, params=params, impersonate="chrome131", timeout=10)
        data = r.json().get("data", {})
        if data is None or "klines" not in data:
            return None
        content_list = data["klines"]
        rows = [item.split(",") for item in content_list]
        df = pd.DataFrame(rows, columns=[
            "日期", "主力净流入-净额", "小单净流入-净额", "中单净流入-净额",
            "大单净流入-净额", "超大单净流入-净额",
            "主力净流入-净占比", "小单净流入-净占比", "中单净流入-净占比",
            "大单净流入-净占比", "超大单净流入-净占比",
            "收盘价", "涨跌幅", "-", "-",
        ])
        return df

    def _get_sector_ranks(self, date_str):
        """获取行业板块涨幅排名，返回 {行业名: rank}"""
        ranks = {}
        try:
            df = ak.stock_board_industry_name_em()
            if df is not None and not df.empty:
                pct_col = None
                for col in ['涨跌幅', '涨幅']:
                    if col in df.columns:
                        pct_col = col
                        break
                name_col = None
                for col in ['板块名称', '名称']:
                    if col in df.columns:
                        name_col = col
                        break
                if pct_col and name_col:
                    df_sorted = df.sort_values(pct_col, ascending=False).reset_index(drop=True)
                    total = len(df_sorted)
                    for idx, row in df_sorted.iterrows():
                        ranks[row[name_col]] = (idx + 1, total)
            print(f"[板块排名] 已加载 {len(ranks)} 个行业")
        except Exception as e:
            print(f"[板块排名] 获取失败: {e}")
        return ranks

    def _score_stock(self, cand, sector_ranks, date_str) -> ScoredStock:
        """对单只候选股进行 12 维打分"""
        code = cand['code']
        name = cand['name']
        lianban = cand['lianban']
        reasons = []

        # ===== A. 连板强度 (15分) =====
        if lianban >= 4:
            s_lb = 15.0
            detail = f"{lianban}连板，超级龙头"
        elif lianban == 3:
            s_lb = 13.0
            detail = f"3连板，强势确认"
        elif lianban == 2:
            s_lb = 10.0
            detail = "2连板，趋势启动"
        else:
            s_lb = 6.0
            detail = "首板，需观察次日溢价"
        reasons.append(RecommendReason('连板强度', s_lb, 15, detail))

        # ===== B. 封板质量 (10分) =====
        fengdan = cand.get('fengdan', 0)
        liutong = cand.get('liutong_mv', 0)
        if liutong > 0 and fengdan > 0:
            fd_ratio = fengdan / liutong
            if fd_ratio > 0.1:
                s_fb = 10.0
                detail = f"封单/流通={fd_ratio:.1%}，死封无量"
            elif fd_ratio > 0.05:
                s_fb = 7.0
                detail = f"封单/流通={fd_ratio:.1%}，封板较实"
            elif fd_ratio > 0.02:
                s_fb = 4.0
                detail = f"封单/流通={fd_ratio:.1%}，封板一般"
            else:
                s_fb = 2.0
                detail = f"封单薄弱，次日可能低开"
        else:
            s_fb = 3.0
            detail = "封单数据缺失"
        reasons.append(RecommendReason('封板质量', s_fb, 10, detail))

        # ===== C. 主力资金 (12分) =====
        s_fund = 0.0
        fund_detail = "数据未获取"
        try:
            flow_df = self._fetch_fund_flow(code)
            if flow_df is not None and not flow_df.empty:
                recent = flow_df.iloc[-1]
                net_ratio = float(recent.get('主力净流入-净占比', 0))
                if net_ratio > 10:
                    s_fund = 12.0
                    fund_detail = f"主力净流入占比{net_ratio:.1f}%，强力吸筹"
                elif net_ratio > 5:
                    s_fund = 9.0
                    fund_detail = f"主力净流入占比{net_ratio:.1f}%，资金关注"
                elif net_ratio > 0:
                    s_fund = 5.0
                    fund_detail = f"主力小幅流入{net_ratio:.1f}%"
                elif net_ratio > -5:
                    s_fund = 2.0
                    fund_detail = f"主力小幅流出{net_ratio:.1f}%"
                else:
                    s_fund = 0.0
                    fund_detail = f"主力大幅流出{net_ratio:.1f}%，⚠️出货嫌疑"
        except Exception:
            s_fund = 3.0
            fund_detail = "资金流向数据获取失败"
        reasons.append(RecommendReason('主力资金', s_fund, 12, fund_detail))

        # ===== D. 技术形态-动量 (8分) =====
        s_mom = 0.0
        mom_detail = "数据未获取"
        try:
            hist_df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                         start_date=(pd.Timestamp(date_str) - pd.Timedelta(days=60)).strftime('%Y%m%d'),
                                         end_date=date_str, adjust="qfq")
            if hist_df is not None and len(hist_df) >= 20:
                close = hist_df['收盘'].values
                vol = hist_df['成交量'].values

                # RSI 14
                import ta
                rsi = ta.momentum.RSIIndicator(pd.Series(close), window=14).rsi().iloc[-1]
                # MACD
                macd_obj = ta.trend.MACD(pd.Series(close))
                macd_diff = macd_obj.macd_diff().iloc[-1]
                # 量比（今日成交量 / 过去5日均量）
                vol_ratio = vol[-1] / (np.mean(vol[-6:-1]) + 1e-8)

                mom_parts = []
                if 40 <= rsi <= 70:
                    s_mom += 3.0
                    mom_parts.append(f"RSI={rsi:.0f}(健康区间)")
                elif rsi > 80:
                    s_mom += 0.0
                    mom_parts.append(f"RSI={rsi:.0f}(超买⚠️)")
                else:
                    s_mom += 1.5
                    mom_parts.append(f"RSI={rsi:.0f}")

                if macd_diff > 0:
                    s_mom += 2.5
                    mom_parts.append("MACD多头")
                else:
                    s_mom += 0.5
                    mom_parts.append("MACD空头")

                if vol_ratio > 2.0:
                    s_mom += 2.5
                    mom_parts.append(f"量比{vol_ratio:.1f}(放量)")
                elif vol_ratio > 1.2:
                    s_mom += 1.5
                    mom_parts.append(f"量比{vol_ratio:.1f}")
                else:
                    s_mom += 0.5
                    mom_parts.append(f"量比{vol_ratio:.1f}(缩量)")

                mom_detail = '，'.join(mom_parts)
        except Exception:
            s_mom = 2.0
            mom_detail = "技术指标计算失败"
        reasons.append(RecommendReason('技术动量', s_mom, 8, mom_detail))

        # ===== E. 技术形态-趋势 (7分) =====
        s_trend = 0.0
        trend_detail = "数据不足"
        try:
            if hist_df is not None and len(hist_df) >= 30:
                close = hist_df['收盘'].values
                ma5 = np.mean(close[-5:])
                ma10 = np.mean(close[-10:])
                ma20 = np.mean(close[-20:])
                cur = close[-1]

                if cur > ma5 > ma10 > ma20:
                    s_trend = 7.0
                    trend_detail = "均线多头排列(5>10>20)，强势趋势"
                elif cur > ma5 and cur > ma20:
                    s_trend = 5.0
                    trend_detail = "站上5日线和20日线"
                elif cur > ma5:
                    s_trend = 3.0
                    trend_detail = "站上5日线，短期向好"
                else:
                    s_trend = 1.0
                    trend_detail = "均线空头，逆势反弹"
        except Exception:
            s_trend = 1.0
        reasons.append(RecommendReason('技术趋势', s_trend, 7, trend_detail))

        # ===== F. 板块共振 (10分) =====
        industry = cand.get('industry', '未知')
        s_sector = 0.0
        sector_detail = f"行业: {industry}"
        if industry in sector_ranks:
            rank, total = sector_ranks[industry]
            pct_rank = rank / total
            if pct_rank <= 0.1:
                s_sector = 10.0
                sector_detail = f"{industry} 今日涨幅排名 {rank}/{total}，板块龙头"
            elif pct_rank <= 0.2:
                s_sector = 8.0
                sector_detail = f"{industry} 排名 {rank}/{total}，板块强势"
            elif pct_rank <= 0.4:
                s_sector = 5.0
                sector_detail = f"{industry} 排名 {rank}/{total}，板块中性"
            else:
                s_sector = 2.0
                sector_detail = f"{industry} 排名 {rank}/{total}，板块偏弱"
        reasons.append(RecommendReason('板块共振', s_sector, 10, sector_detail))

        # ===== G. 社区情绪 (10分) =====
        sent_result = self.sentiment_engine.score(code)
        # 缩放到10分制 (原15分制)
        s_sent = min(sent_result.total_score * 10.0 / 15.0, 10.0)
        sent_tags = sent_result.tags
        if sent_tags:
            sent_detail = f"东财: {', '.join(sent_tags)}"
        else:
            sent_detail = "未上东财热榜/飙升榜"
        reasons.append(RecommendReason('社区情绪', s_sent, 10, sent_detail))

        # ===== H. 龙虎榜席位 (8分) =====
        s_lhb = 4.0  # 默认中性
        lhb_detail = "今日无龙虎榜数据"
        # 龙虎榜数据不一定每天有，做防御处理
        try:
            lhb_df = ak.stock_lhb_detail_em(
                start_date=date_str, end_date=date_str
            )
            if lhb_df is not None and not lhb_df.empty:
                code_col = None
                for c in ['代码', '股票代码']:
                    if c in lhb_df.columns:
                        code_col = c
                        break
                if code_col:
                    stock_lhb = lhb_df[lhb_df[code_col].astype(str).str.zfill(6) == code]
                    if not stock_lhb.empty:
                        # 检查是否有拉萨席位
                        lasa_count = 0
                        for _, r in stock_lhb.iterrows():
                            seat = str(r.get('营业部名称', ''))
                            if '拉萨' in seat:
                                lasa_count += 1
                        if lasa_count >= 2:
                            s_lhb = 0.0
                            lhb_detail = f"龙虎榜惊现 {lasa_count} 家拉萨席位，⚠️散户接盘高危"
                        elif lasa_count == 1:
                            s_lhb = 3.0
                            lhb_detail = "龙虎榜有1家拉萨席位，需警惕"
                        else:
                            s_lhb = 8.0
                            lhb_detail = "龙虎榜无拉萨席位，游资合力良好"
        except Exception:
            pass
        reasons.append(RecommendReason('龙虎榜席位', s_lhb, 8, lhb_detail))

        # ===== I. 换手率异动 (5分，反向) =====
        turnover = cand.get('turnover_rate', 0)
        s_turn = 3.0
        turn_detail = f"换手率 {turnover:.1f}%"
        if turnover > 25:
            s_turn = 0.0
            turn_detail = f"换手率 {turnover:.1f}%，⚠️换手过高（放量出货风险）"
        elif turnover > 15:
            s_turn = 2.0
            turn_detail = f"换手率 {turnover:.1f}%，偏高需关注"
        elif 5 <= turnover <= 15:
            s_turn = 5.0
            turn_detail = f"换手率 {turnover:.1f}%，健康区间"
        elif turnover < 3 and turnover > 0:
            s_turn = 4.0
            turn_detail = f"换手率 {turnover:.1f}%，缩量（一字板）"
        reasons.append(RecommendReason('换手率', s_turn, 5, turn_detail))

        # ===== J. 情绪资金背离 (5分，反向扣分) =====
        s_div = 5.0  # 默认无背离=满分
        div_detail = "未检测到情绪-资金背离"
        if sent_result.total_score >= 10 and s_fund <= 2:
            s_div = 0.0
            div_detail = "⚠️社区极度狂热但主力资金流出，疑似杀猪盘！"
        elif sent_result.total_score >= 8 and s_fund <= 3:
            s_div = 2.0
            div_detail = "⚠️社区热度高+主力流出，注意风险"
        reasons.append(RecommendReason('情绪背离', s_div, 5, div_detail))

        # ===== K. 市值适配 (5分) =====
        liutong_yi = liutong / 1e8 if liutong > 0 else 0
        if 20 <= liutong_yi <= 200:
            s_mcap = 5.0
            mcap_detail = f"流通市值 {liutong_yi:.0f}亿，游资偏好区间"
        elif 10 <= liutong_yi < 20:
            s_mcap = 3.0
            mcap_detail = f"流通市值 {liutong_yi:.0f}亿，偏小但可操作"
        elif 200 < liutong_yi <= 500:
            s_mcap = 3.0
            mcap_detail = f"流通市值 {liutong_yi:.0f}亿，偏大需更多资金推动"
        elif liutong_yi > 500:
            s_mcap = 1.0
            mcap_detail = f"流通市值 {liutong_yi:.0f}亿，大象股难连板"
        elif liutong_yi < 5 and liutong_yi > 0:
            s_mcap = 1.0
            mcap_detail = f"流通市值 {liutong_yi:.1f}亿，⚠️微盘股风险高"
        else:
            s_mcap = 2.0
            mcap_detail = "市值数据缺失"
        reasons.append(RecommendReason('市值适配', s_mcap, 5, mcap_detail))

        # ===== L. 安全评级 (5分) =====
        s_safe = 5.0
        safe_detail = "通过安全检查"
        if 'ST' in name.upper():
            s_safe = 0.0
            safe_detail = "ST股，已排除"
        reasons.append(RecommendReason('安全评级', s_safe, 5, safe_detail))

        # ===== 汇总 =====
        total = (s_lb + s_fb + s_fund + s_mom + s_trend +
                 s_sector + s_sent + s_lhb + s_turn + s_div + s_mcap + s_safe)

        stock = ScoredStock(
            code=code, name=name, industry=industry,
            lianban=lianban, total_score=total, reasons=reasons,
            score_lianban=s_lb, score_fengban=s_fb, score_fund=s_fund,
            score_momentum=s_mom, score_trend=s_trend, score_sector=s_sector,
            score_sentiment=s_sent, score_lhb=s_lhb, score_turnover=s_turn,
            score_divergence=s_div, score_mcap=s_mcap, score_safety=s_safe,
            sentiment_tags=sent_tags,
            close_price=cand.get('close', 0),
            pct_change=cand.get('pct_change', 0),
            turnover_rate=turnover,
            circulating_mcap=liutong_yi,
        )
        return stock


if __name__ == "__main__":
    engine = DailyScannerEngine(board_filter='main')
    results, market = engine.scan()

    print(f"\n{'='*60}")
    print(f"推荐列表 ({market.label_cn} 市场)")
    print(f"{'='*60}")
    for i, s in enumerate(results):
        print(f"\n#{i+1} {s.code} {s.name} [{s.industry}] — {s.total_score:.1f}分 ({s.lianban}连板)")
        for r in s.top_reasons():
            print(f"    {r}")
        warnings = s.risk_warnings()
        if warnings:
            print(f"    ⚠️ 风险提示:")
            for w in warnings:
                print(f"    {w}")

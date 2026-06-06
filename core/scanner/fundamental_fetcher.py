import akshare as ak
import pandas as pd
import numpy as np

class FundamentalFetcher:
    def __init__(self):
        pass

    def get_financial_safety_score(self, symbol, name):
        """
        排雷逻辑（宽容/题材优先）：
        只剔除 ST、*ST 股。允许财报亏损但题材纯正的妖股进入。
        返回 1.0 表示安全，0.0 表示高危垃圾股(ST)。
        """
        if "ST" in str(name).upper():
            return 0.0
        return 1.0

    def turnover_ratio_check(self, symbol, current_turnover, ma5_turnover, is_limit_up):
        """
        换手率异动监控 (Abnormal Turnover)
        如果当日换手率超过了过去 5 天平均换手率的 3 倍以上，但股价涨幅却无法封死涨停（放量滞涨）。
        返回 True 表示存在放量滞涨（出货嫌疑）。
        """
        if ma5_turnover <= 0:
            return False
            
        if current_turnover > 3 * ma5_turnover and not is_limit_up:
            print(f"[警戒] {symbol} 换手率飙升至 {current_turnover}% (>3倍均量)，且未封死涨停，疑似大户倒货！")
            return True
        return False

    def get_lhb_seat_quality(self, symbol, date_str):
        “””
        龙虎榜”拉萨天团”权重（Seat Quality Analysis）
        买入前五中，包含”拉萨”的席位越多，质量分越低。
        返回 [-1.0, 1.0] 的分数。-1 代表全是大本营散户，1 代表全是顶级游资。
        “””
        code = str(symbol).zfill(6)
        try:
            lhb_df = ak.stock_lhb_detail_em(date=date_str)
            if lhb_df is None or lhb_df.empty:
                return 0.0  # 当日无龙虎榜数据，返回中性
            stock_lhb = lhb_df[lhb_df['代码'] == code]
            if stock_lhb.empty:
                return 0.0
            buy_seats = stock_lhb[stock_lhb['类型'].str.contains('买入', na=False)]
            if buy_seats.empty:
                return 0.0
            lhasa_count = buy_seats['营业部名称'].str.contains('拉萨', na=False).sum()
            if lhasa_count > 0:
                print(f”[龙虎榜] {code} 买入前五出现 {lhasa_count} 家拉萨席位，降低预期评分。”)
                return -0.5 * min(lhasa_count, 2)
            return 0.5
        except Exception:
            return 0.0  # 获取失败返回中性，不产生误导信号

    def get_stock_info(self, symbol):
        """
        获取单只股票的基本财务信息（总市值、流通市值、市盈率等）。
        仅支持主板股票 (60xxxx, 00xxxx)。
        """
        code = str(symbol).zfill(6)
        if not (code.startswith('60') or code.startswith('00')):
            return None
            
        try:
            info_df = ak.stock_individual_info_em(symbol=code)
            info_dict = dict(zip(info_df['item'], info_df['value']))
            
            return {
                'symbol': code,
                'total_market_cap': info_dict.get('总市值', 0),
                'circulating_market_cap': info_dict.get('流通市值', 0),
                'pe_ratio': info_dict.get('市盈率-动态', 0),
                'industry': info_dict.get('行业', '未知')
            }
        except Exception as e:
            return None

    def get_main_money_flow(self, symbol):
        """
        获取主力资金流向。用于量价背离检查。
        """
        code = str(symbol).zfill(6)
        try:
            flow_df = ak.stock_individual_fund_flow(stock=code, market="sh" if code.startswith('60') else "sz")
            if not flow_df.empty:
                recent_flow = flow_df.iloc[-1]
                return {
                    'date': recent_flow['日期'],
                    'main_net_inflow': recent_flow.get('主力净流入-净额', 0),
                    'main_net_inflow_ratio': recent_flow.get('主力净流入-净占比', 0)
                }
            return None
        except Exception:
            return None

import akshare as ak
import pandas as pd
import datetime

class SectorFetcher:
    def __init__(self):
        self.sector_cache = {}

    def get_stock_sector(self, symbol):
        """
        获取股票所属行业板块。
        通过 akshare 获取个股信息中的行业字段。
        """
        code = str(symbol).zfill(6)
        if code in self.sector_cache:
            return self.sector_cache[code]
        try:
            info = ak.stock_individual_info_em(symbol=code)
            info_dict = dict(zip(info['item'], info['value']))
            sector = info_dict.get('行业', '未知')
            self.sector_cache[code] = sector
            return sector
        except Exception:
            return "未知"

    def calculate_divergence(self, symbol, stock_pct_change, date_str=None):
        """
        计算个股与板块的偏离度 (Divergence = Delta Price_stock - Delta Price_sector)
        :param symbol: 股票代码
        :param stock_pct_change: 个股当日涨幅 (百分比，如 5.0 表示 5%)
        :return: divergence (float), sector_pct_change (float)
        """
        code = str(symbol).zfill(6)
        try:
            sector_name = self.get_stock_sector(code)
            if sector_name == "未知":
                return 0.0, 0.0
            # 尝试获取板块指数近期行情
            if date_str is None:
                date_str = datetime.date.today().strftime('%Y%m%d')
            board_df = ak.stock_board_industry_hist_em(
                symbol=sector_name, period="daily",
                start_date=date_str, end_date=date_str,
            )
            if board_df is not None and not board_df.empty:
                sector_pct = float(board_df.iloc[-1].get('涨跌幅', 0))
            else:
                sector_pct = 0.0
            divergence = stock_pct_change - sector_pct
            return divergence, sector_pct
        except Exception:
            return 0.0, 0.0

if __name__ == "__main__":
    fetcher = SectorFetcher()
    div, sec_pct = fetcher.calculate_divergence("000001", 6.5)
    print(f"Sector Pct: {sec_pct}%, Divergence: {div}%")

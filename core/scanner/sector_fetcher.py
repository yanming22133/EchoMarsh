import akshare as ak
import pandas as pd
import datetime

class SectorFetcher:
    def __init__(self):
        self.sector_cache = {}

    def get_stock_sector(self, symbol):
        """
        获取股票所属板块。此处简化返回一个默认板块用于演示。
        实际可通过 ak.stock_board_industry_name_em 匹配。
        """
        return "所属行业板块"

    def calculate_divergence(self, symbol, stock_pct_change, date_str=None):
        """
        计算个股与板块的偏离度 (Divergence = Delta Price_stock - Delta Price_sector)
        :param symbol: 股票代码
        :param stock_pct_change: 个股当日涨幅 (百分比，如 5.0 表示 5%)
        :return: divergence (float), sector_pct_change (float)
        """
        code = str(symbol).zfill(6)
        
        try:
            # 简化演示：假设获取到了板块的指数涨跌幅
            # 实际中应获取板块指数的收盘价来计算
            # sector_name = self.get_stock_sector(code)
            # board_df = ak.stock_board_industry_hist_em(symbol=sector_name)
            
            # 模拟：板块走势通常较弱，假设当天板块跌了 1%
            mock_sector_pct_change = -1.0 
            
            divergence = stock_pct_change - mock_sector_pct_change
            
            return divergence, mock_sector_pct_change
            
        except Exception as e:
            print(f"Error calculating divergence for {code}: {e}")
            return 0.0, 0.0

if __name__ == "__main__":
    fetcher = SectorFetcher()
    div, sec_pct = fetcher.calculate_divergence("000001", 6.5)
    print(f"Sector Pct: {sec_pct}%, Divergence: {div}%")

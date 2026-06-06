import os
import pathlib
import datetime
import pandas as pd
import akshare as ak
import time

# 绕过系统级代理，防止 akshare 请求国内数据源时被拦截
os.environ["NO_PROXY"] = "*"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

def get_recent_trade_dates(n=10):
    """获取最近 N 个交易日"""
    try:
        trade_dates_df = ak.tool_trade_date_hist_sina()
        trade_dates = pd.to_datetime(trade_dates_df['trade_date']).dt.date
        today = datetime.date.today()
        recent_dates = trade_dates[trade_dates <= today].tail(n).tolist()
        return recent_dates
    except Exception as e:
        print(f"Error fetching trade dates: {e}")
        return []

def get_top_zt_stocks(date_str):
    """获取指定日期的连板前5名"""
    try:
        zt_pool = ak.stock_zt_pool_em(date=date_str)
        if zt_pool.empty or '连板数' not in zt_pool.columns:
            return pd.DataFrame()
        # 排序并获取前5
        top_5 = zt_pool.sort_values(by='连板数', ascending=False).head(5)
        return top_5
    except Exception as e:
        print(f"Error fetching ZT pool for {date_str}: {e}")
        return pd.DataFrame()

def fetch_stock_data(symbol, start_date_str, output_dir):
    """
    获取指定股票在起步当天的1分钟数据，计算竞价量比并保存。
    """
    print(f"Fetching data for {symbol} on {start_date_str}")
    # 提取六位代码
    code = str(symbol).zfill(6)
    
    try:
        # 获取日线数据以获取昨日成交量
        # 我们需要获取start_date_str之前的一个交易日的成交量
        end_date = datetime.datetime.strptime(start_date_str, "%Y%m%d").date()
        start_date = end_date - datetime.timedelta(days=30)
        
        daily_df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date.strftime("%Y%m%d"), end_date=start_date_str, adjust="qfq")
        if daily_df.empty or len(daily_df) < 2:
            print(f"Not enough daily data for {symbol}")
            return
            
        # 最后一行是start_date_str当天，倒数第二行是昨天
        yesterday_data = daily_df.iloc[-2]
        today_data = daily_df.iloc[-1]
        
        yesterday_total_volume = yesterday_data['成交量']
        open_price = today_data['开盘']
        
        # 获取1分钟数据
        start_time = f"{start_date_str} 09:00:00"
        end_time = f"{start_date_str} 15:30:00"
        
        # stock_zh_a_hist_min_em 获取分钟级历史数据
        min_df = ak.stock_zh_a_hist_min_em(symbol=code, start_date=start_time, end_date=end_time, period='1', adjust='qfq')
        
        if min_df.empty:
            print(f"No 1-min data for {symbol} on {start_date_str}")
            return
            
        # 获取09:30的成交量作为集合竞价成交量（AkShare的分钟线特性）
        first_min_data = min_df.iloc[0]
        auction_volume = first_min_data['成交量']
        auction_amount = first_min_data['成交额']
        
        # 计算竞价量比 (Ratio = Auction_Volume / Yesterday_Total_Volume)
        if yesterday_total_volume > 0:
            ratio = auction_volume / yesterday_total_volume
        else:
            ratio = 0
            
        print(f"[{code}] {start_date_str} - 竞价成交量: {auction_volume}, 昨日成交量: {yesterday_total_volume}, 竞价量比: {ratio:.4f}")
        
        # 将特征加入到1分钟数据中
        min_df['auction_volume'] = auction_volume
        min_df['yesterday_total_volume'] = yesterday_total_volume
        min_df['auction_ratio'] = ratio
        min_df['auction_open_price'] = open_price
        
        # 确保使用 os.path.join 构建路径
        save_path = os.path.join(output_dir, f"{code}_{start_date_str}.csv")
        min_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"Saved to {save_path}")
        
    except Exception as e:
        print(f"Error processing {symbol}: {e}")

def main():
    # 强制规范化路径
    base_dir = pathlib.Path(__file__).resolve().parent.parent.parent
    raw_data_dir = base_dir / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    print("EchoMarsh 交易系统 - 数据捕获实验室 (Data Fetcher)")
    print("="*60)
    
    trade_dates = get_recent_trade_dates(10)
    print(f"Found {len(trade_dates)} recent trade dates.")
    
    # 获取全市场交易日历，用于计算"起步当天"
    all_trade_dates_df = ak.tool_trade_date_hist_sina()
    all_trade_dates = pd.to_datetime(all_trade_dates_df['trade_date']).dt.date.tolist()
    
    for target_date in trade_dates:
        target_date_str = target_date.strftime("%Y%m%d")
        print(f"\n>> 扫描日期: {target_date_str}")
        
        top_stocks = get_top_zt_stocks(target_date_str)
        if top_stocks.empty:
            continue
            
        for index, row in top_stocks.iterrows():
            code = row['代码']
            name = row['名称']
            lianban = row.get('连板数', 1)
            
            # 策略约束：目前只考虑基础账户（主板），过滤掉创业板(300)、科创板(688)、北交所(8/4)
            if not (str(code).startswith('60') or str(code).startswith('00')):
                print(f"Skipping non-main-board stock: {name} ({code})")
                continue
            
            # 计算起步当天：往前回推 N-1 个交易日
            try:
                date_idx = all_trade_dates.index(target_date)
                start_date_idx = date_idx - (lianban - 1)
                if start_date_idx >= 0:
                    start_date = all_trade_dates[start_date_idx]
                    start_date_str_fmt = start_date.strftime("%Y%m%d")
                    
                    print(f"Target: {name} ({code}), 当前连板: {lianban}, 起步日期: {start_date_str_fmt}")
                    fetch_stock_data(code, start_date_str_fmt, str(raw_data_dir))
                    time.sleep(1)  # 防止请求过频被封IP
            except ValueError:
                pass
                
if __name__ == "__main__":
    main()

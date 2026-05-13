import os
import pandas as pd
import numpy as np

def generate_mock_data(output_dir, symbol="000001", date="20260512"):
    """
    生成一份伪造的1分钟级别的K线数据用于测试数据处理管道。
    包含240分钟（一个完整交易日）的数据。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成时间序列 09:30 - 11:30, 13:00 - 15:00
    times1 = pd.date_range(f"{date} 09:30:00", f"{date} 11:30:00", freq="1min")
    times2 = pd.date_range(f"{date} 13:00:00", f"{date} 15:00:00", freq="1min")
    times = times1.append(times2)
    
    n_samples = len(times)
    
    # 随机生成开高低收等数据
    np.random.seed(42)
    start_price = 10.0
    returns = np.random.normal(0, 0.002, n_samples)
    prices = start_price * np.exp(np.cumsum(returns))
    
    highs = prices * (1 + np.abs(np.random.normal(0, 0.001, n_samples)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.001, n_samples)))
    opens = prices * (1 + np.random.normal(0, 0.0005, n_samples))
    closes = prices
    
    # 保证高低价逻辑
    highs = np.maximum.reduce([opens, closes, highs])
    lows = np.minimum.reduce([opens, closes, lows])
    
    volumes = np.random.randint(1000, 50000, n_samples)
    amounts = volumes * 100 * closes # 假设1手=100股
    
    df = pd.DataFrame({
        '时间': times,
        '开盘': opens,
        '收盘': closes,
        '最高': highs,
        '最低': lows,
        '成交量': volumes,
        '成交额': amounts,
        'auction_volume': 150000,
        'yesterday_total_volume': 5000000,
        'auction_ratio': 150000 / 5000000,
        'auction_open_price': start_price
    })
    
    file_path = os.path.join(output_dir, f"{symbol}_{date}.csv")
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"Mock data saved to {file_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    generate_mock_data(raw_dir)

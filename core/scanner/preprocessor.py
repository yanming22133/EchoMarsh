import pandas as pd
import numpy as np
import ta

class Preprocessor:
    def __init__(self, sequence_length=60, future_window=15):
        self.sequence_length = sequence_length
        self.future_window = future_window

        # 特征列（22 维 TA 因子 + 基础量价 + 竞价特征）
        self.feature_cols = [
            '开盘', '最高', '最低', '收盘', '成交量', '成交额',
            'return', 'log_return', 'high_low_spread', 'volume_change',
            'rsi', 'tsi', 'mfi', 'macd', 'macd_diff', 'adx',
            'bb_w', 'bb_pband', 'atr', 'vwap_dev', 'cmf', 'auction_ratio'
        ]

    def add_ta_features(self, df):
        df = df.ffill().bfill()
        try:
            df['rsi']      = ta.momentum.RSIIndicator(close=df['收盘'], window=14).rsi()
            df['tsi']      = ta.momentum.TSIIndicator(close=df['收盘']).tsi()
            df['mfi']      = ta.volume.MFIIndicator(high=df['最高'], low=df['最低'],
                                                    close=df['收盘'], volume=df['成交量']).money_flow_index()
            macd           = ta.trend.MACD(close=df['收盘'])
            df['macd']     = macd.macd()
            df['macd_diff']= macd.macd_diff()
            df['adx']      = ta.trend.ADXIndicator(high=df['最高'], low=df['最低'],
                                                   close=df['收盘']).adx()
            bb             = ta.volatility.BollingerBands(close=df['收盘'], window=20, window_dev=2)
            df['bb_w']     = bb.bollinger_wband()
            df['bb_pband'] = bb.bollinger_pband()
            df['atr']      = ta.volatility.AverageTrueRange(high=df['最高'], low=df['最低'],
                                                            close=df['收盘']).average_true_range()
            df['vwap']     = ta.volume.VolumeWeightedAveragePrice(
                high=df['最高'], low=df['最低'], close=df['收盘'], volume=df['成交量']
            ).volume_weighted_average_price()
            df['vwap_dev'] = (df['收盘'] - df['vwap']) / (df['vwap'] + 1e-8)
            df['cmf']      = ta.volume.ChaikinMoneyFlowIndicator(
                high=df['最高'], low=df['最低'], close=df['收盘'], volume=df['成交量']
            ).chaikin_money_flow()
        except Exception as e:
            print(f"TA features error: {e}")
        return df

    def _rolling_zscore_normalize(self, df, cols, window=120):
        """
        滚动窗口 Z-Score 归一化 —— 关键：只使用过去数据的统计量，杜绝信息泄漏。
        window=120 (分钟) 即用过去 2 小时的分布来归一化当前时刻
        """
        for col in cols:
            if col not in df.columns:
                df[f'{col}_norm'] = 0.0
                continue
            roll_mean = df[col].rolling(window=window, min_periods=10).mean()
            roll_std  = df[col].rolling(window=window, min_periods=10).std()
            df[f'{col}_norm'] = (df[col] - roll_mean) / (roll_std + 1e-8)
        return df

    def _build_label(self, df):
        """
        构建标签：严格使用 t+1 到 t+future_window 的未来最大涨幅
        这里使用 shift(-1) 后再 rolling，确保 t 时刻用的是 t+1 之后的数据
        """
        future_max_high = (
            df['最高']
            .shift(-1)                     # 从 t+1 开始
            .rolling(window=self.future_window, min_periods=1)
            .max()
            .shift(-(self.future_window - 1))  # 对齐回当前时刻
        )
        df['target_max_return_15m'] = (future_max_high - df['收盘']) / (df['收盘'] + 1e-8)
        return df

    def process_file(self, file_path):
        try:
            df = pd.read_csv(file_path)
            if df.empty or len(df) < self.sequence_length + self.future_window:
                return None

            df['时间'] = pd.to_datetime(df['时间'])
            df = df.sort_values('时间').reset_index(drop=True)

            # 基础差分特征
            df['return']          = df['收盘'].pct_change()
            df['log_return']      = np.log(df['收盘'] / (df['收盘'].shift(1) + 1e-8))
            df['high_low_spread'] = (df['最高'] - df['最低']) / (df['开盘'] + 1e-8)
            df['volume_change']   = df['成交量'].pct_change()

            # TA 因子
            df = self.add_ta_features(df)

            # 补全可能缺失的列
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0.0

            # 【已修复】滚动窗口 Z-Score 归一化（无信息泄漏）
            df = self._rolling_zscore_normalize(df, self.feature_cols, window=120)

            # 【已修复】严格无前视偏差的标签计算
            df = self._build_label(df)

            # 丢弃 NaN（TA 指标计算需要预热期）
            df = df.dropna().reset_index(drop=True)

            if len(df) < self.sequence_length:
                return None

            return df

        except Exception as e:
            print(f"Error preprocessing {file_path}: {e}")
            return None

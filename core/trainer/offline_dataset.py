"""
EchoMarsh 离线股票数据集加载器 (PyTorch) — 日线版
===============================================
用于读取按"每只股票一个文件"存储的历史日线 CSV 数据。
添加完整的 TA 特征工程 + 真实元特征，并修复训练阻塞 bug。
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ========== 特征定义 ==========

# 日线 22 维特征（对齐模型 ts_feature_dim=22）
FEATURE_COLS = [
    # 基础量价 (6)
    '开盘价', '最高价', '最低价', '收盘价', '成交量（股）', '成交额（元）',
    # 衍生收益率 (3)
    '涨幅%', '3日涨幅%', '10日涨幅%',
    # 流动性与波动 (3)
    '换手率', '振幅%', '量比',
    # 均线乖离率 (5) — 相对于收盘价的偏离%
    'dev_ma5', 'dev_ma10', 'dev_ma20', 'dev_ma30', 'dev_ma60',
    # TA 技术指标 (5)
    'rsi_14', 'macd', 'macd_diff', 'bb_pband', 'atr_14_norm',
]

# 需要做 Z-Score 归一化的特征（指标类特征本身有界，不重复归一化）
NORM_COLS = [
    '开盘价', '最高价', '最低价', '收盘价',
    '成交量（股）', '成交额（元）',
    '换手率', '振幅%',
    'atr_14_norm',
]


def add_daily_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """在日线 DataFrame 上计算 TA 技术指标"""
    close = df['收盘价'].values.astype(np.float64)
    high = df['最高价'].values.astype(np.float64)
    low = df['最低价'].values.astype(np.float64)

    eps = 1e-10

    # --- RSI 14 ---
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[:14])
    avg_loss[13] = np.mean(loss[:14])
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i - 1] * 13 + loss[i]) / 14
    rs = avg_gain / (avg_loss + eps)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    df['macd'] = ema12 - ema26
    df['macd_signal'] = _ema(df['macd'].values, 9)
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # --- Bollinger Band %b ---
    ma20 = _sma(close, 20)
    std20 = _rolling_std(close, 20)
    df['bb_pband'] = (close - ma20) / (2 * std20 + eps) + 0.5

    # --- ATR 14 (归一化到收盘价) ---
    tr = np.maximum(high - low,
                    np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    df['atr_14'] = _ema(tr, 14)
    df['atr_14_norm'] = df['atr_14'] / (close + eps) * 100

    return df


def _ema(x: np.ndarray, period: int) -> np.ndarray:
    """
    指数移动平均 — 使用 pandas 实现（稳健处理前导 NaN）。
    NaN 值不会传播：从第一个有效值开始计算。
    """
    return pd.Series(x).ewm(span=period, adjust=False, min_periods=1).mean().to_numpy()


def _sma(x: np.ndarray, period: int) -> np.ndarray:
    """简单移动平均 — pandas rolling 实现，min_periods=1 避免全NaN"""
    return pd.Series(x).rolling(window=period, min_periods=1).mean().to_numpy()


def _rolling_std(x: np.ndarray, period: int) -> np.ndarray:
    """滚动标准差（总体标准差 ddof=0）"""
    return pd.Series(x).rolling(window=period, min_periods=1).std(ddof=0).to_numpy()


class OfflineStockDataset(Dataset):
    """
    EchoMarsh 离线股票数据集 (日线版)

    用法:
        ds = OfflineStockDataset(
            data_dir="data/offline/前复权",
            seq_len=120,          # 日线用 120 天 ≈ 6 个月
            pred_len=5,           # 预测未来 5 日收益
            is_train=True,
            start_date='2010-01-01',
            end_date='2024-01-01',
            max_files=None,       # None=全部, 数字=限制文件数
        )
        loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)
    """

    def __init__(self, data_dir, seq_len=120, pred_len=5, is_train=True,
                 train_ratio=0.8, start_date=None, end_date=None,
                 max_files=None):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.is_train = is_train
        self.train_ratio = train_ratio
        self.max_files = max_files

        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None

        self.files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if not self.files:
            print(f"[警告] 在 {data_dir} 未找到 CSV 文件.")

        # 裁剪测试用文件数
        if self.max_files is not None:
            self.files = self.files[:self.max_files]

        self.samples = []   # [(pre_normed_feats, meta_arr, label), ...]
        self.scaler = StandardScaler()
        self._scaler_fitted = False

        print(f"正在预处理 {len(self.files)} 个文件 (seq_len={seq_len}, pred_len={pred_len})...")
        self._build_dataset()

    def _build_dataset(self):
        """逐个文件处理，构建样本"""
        all_feature_buffer = []  # 用于拟合 scaler 的特征缓冲

        for file_idx, file in enumerate(self.files):
            try:
                try:
                    df = pd.read_csv(file, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file, encoding='gbk')
                    except UnicodeDecodeError:
                        df = pd.read_csv(file, encoding='utf-8')

                if '收盘价' not in df.columns or '日期' not in df.columns:
                    continue
                if len(df) < self.seq_len + self.pred_len + 60:  # +60 给 TA 预热
                    continue

                df['日期'] = pd.to_datetime(df['日期'])
                df = df.sort_values('日期').reset_index(drop=True)

                # 时间过滤
                if self.start_date:
                    df = df[df['日期'] >= self.start_date]
                if self.end_date:
                    df = df[df['日期'] <= self.end_date]
                if len(df) < self.seq_len + self.pred_len + 30:
                    continue

                # 缺失值处理
                df = df.ffill().bfill().fillna(0)

                # === 特征工程 ===
                close = df['收盘价'].values
                high = df['最高价'].values
                low = df['最低价'].values
                vol = df['成交量（股）'].values
                eps = 1e-8

                # 均线乖离率 (deviation from MA, 百分比%)
                for ma_name, ma_period in [('dev_ma5', 5), ('dev_ma10', 10),
                                           ('dev_ma20', 20), ('dev_ma30', 30),
                                           ('dev_ma60', 60)]:
                    ma = _sma(close, ma_period)
                    df[ma_name] = (close - ma) / (ma + eps) * 100

                # TA 技术指标
                df = add_daily_ta_features(df)

                # 补齐缺失特征列
                for col in FEATURE_COLS:
                    if col not in df.columns:
                        df[col] = 0.0

                # TA 指标计算后有前导 NaN（RSI/MACD 预热期），前向填充后补 0
                df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill().fillna(0)

                # === 标签构建 (未来 pred_len 日收益率, 百分比%) ===
                future_price = df['收盘价'].shift(-self.pred_len)
                df['Target_Return'] = (future_price - df['收盘价']) / (df['收盘价'] + eps) * 100
                df = df.dropna(subset=['Target_Return'])

                if len(df) < self.seq_len:
                    continue

                # 收集特征 (用于 scaler 拟合)
                feat_raw = df[FEATURE_COLS].values.astype(np.float32)

                # 去 nan/inf
                valid_mask = ~(np.isnan(feat_raw).any(axis=1) | np.isinf(feat_raw).any(axis=1))
                feat_raw = feat_raw[valid_mask]
                targets = df['Target_Return'].values[valid_mask]

                if len(feat_raw) < self.seq_len:
                    continue

                # 收集用于 scaler 拟合的样本 (每个文件采样 500 行)
                if self.is_train:
                    n_sample = min(500, len(feat_raw))
                    indices = np.linspace(0, len(feat_raw) - 1, n_sample, dtype=int)
                    all_feature_buffer.append(feat_raw[indices])

                # === Meta 特征提取 (取最新时间点的数据) ===
                if '总市值（元）' in df.columns:
                    log_mcap = np.log1p(float(df['总市值（元）'].iloc[-1]))
                else:
                    log_mcap = 0.0
                if '滚动市盈率' in df.columns:
                    pe = float(df['滚动市盈率'].iloc[-1])
                    pe = np.clip(pe, 0, 200) / 50.0
                else:
                    pe = 0.0
                if '市净率' in df.columns:
                    pb = float(df['市净率'].iloc[-1])
                    pb = np.clip(pb, 0, 20) / 5.0
                else:
                    pb = 0.0
                turnover_vals = df['换手率'].values[-5:] if '换手率' in df.columns else np.zeros(5)
                turnover_ma5 = float(np.nanmean(turnover_vals)) / 10.0
                if '流通市值（元）' in df.columns and '总市值（元）' in df.columns:
                    circ_ratio = float(df['流通市值（元）'].iloc[-1]) / (float(df['总市值（元）'].iloc[-1]) + eps)
                else:
                    circ_ratio = 0.5
                ret_5d = float(df['涨幅%'].values[-5:].sum()) / 20.0
                limit_up = float(df['是否涨停'].iloc[-1] == '是') if '是否涨停' in df.columns else 0.0

                meta_arr = np.array([
                    log_mcap * 0.1,
                    pe,
                    pb,
                    turnover_ma5,
                    circ_ratio,
                    ret_5d,
                    limit_up,
                ], dtype=np.float32)

                # === 滑动窗口生成样本 ===
                for i in range(len(feat_raw) - self.seq_len + 1):
                    x = feat_raw[i:i + self.seq_len]
                    y = targets[i + self.seq_len - 1]
                    if np.isnan(x).any() or np.isinf(x).any() or np.isnan(y) or np.isinf(y):
                        continue
                    self.samples.append((x, meta_arr.copy(), y))

            except Exception as e:
                print(f"处理文件 {file} 出错: {e}")
                continue

            if (file_idx + 1) % 200 == 0:
                print(f"  已处理 {file_idx+1}/{len(self.files)} 个文件, 样本数: {len(self.samples)}")

        # === 拟合 StandardScaler ===
        if self.is_train and all_feature_buffer:
            concat = np.vstack(all_feature_buffer)
            norm_indices = [i for i, c in enumerate(FEATURE_COLS) if c in NORM_COLS]
            self.scaler.fit(concat[:, norm_indices])
            self._scaler_fitted = True
            print(f"  StandardScaler 拟合完成 (基于 {len(concat)} 行, {len(norm_indices)} 个特征)")

        # === 预归一化所有样本 ===
        if self._scaler_fitted:
            norm_indices = [i for i, c in enumerate(FEATURE_COLS) if c in NORM_COLS]
            pre_normed = []
            for x, meta, y in self.samples:
                x_norm = x.copy().astype(np.float32)
                x_norm[:, norm_indices] = self.scaler.transform(x_norm[:, norm_indices])
                pre_normed.append((x_norm.astype(np.float16), meta, y))
            self.samples = pre_normed

        print(f"[{'Train' if self.is_train else 'Val'}] 数据集构建完成: {len(self.samples)} 个样本")

    def set_scaler(self, scaler):
        """验证集使用训练集的标准化器"""
        self.scaler = scaler
        self._scaler_fitted = True
        norm_indices = [i for i, c in enumerate(FEATURE_COLS) if c in NORM_COLS]
        pre_normed = []
        for x, meta, y in self.samples:
            x_norm = x.copy().astype(np.float32)
            x_norm[:, norm_indices] = self.scaler.transform(x_norm[:, norm_indices])
            pre_normed.append((x_norm.astype(np.float16), meta, y))
        self.samples = pre_normed
        print(f"[Val] 使用训练集 scaler 重归一化完成")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, meta, y = self.samples[idx]
        return (
            torch.from_numpy(x.astype(np.float32)),  # [seq_len, 22]
            torch.from_numpy(meta),                   # [7]
            torch.tensor(y, dtype=torch.float32),     # scalar
        )


def get_dataloaders(data_dir, batch_size=32, seq_len=120, pred_len=5,
                    train_ratio=0.8, start_date=None, end_date=None,
                    num_workers=2, max_files=None):
    """一键创建 train/val DataLoader"""
    train_ds = OfflineStockDataset(
        data_dir, seq_len=seq_len, pred_len=pred_len,
        is_train=True, train_ratio=train_ratio,
        start_date=start_date, end_date=end_date,
        max_files=max_files,
    )
    val_ds = OfflineStockDataset(
        data_dir, seq_len=seq_len, pred_len=pred_len,
        is_train=False, train_ratio=train_ratio,
        start_date=start_date, end_date=end_date,
        max_files=max_files,
    )
    val_ds.set_scaler(train_ds.scaler)

    pin = (num_workers > 0)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    ) if len(train_ds) else None
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    ) if len(val_ds) else None

    return train_loader, val_loader


class OfflineStockDatasetFromFiles(Dataset):
    """
    支持直接传入文件列表构建数据集（用于 Walk-Forward 按窗口切分）。
    使用训练集的 scaler 做归一化。
    """
    def __init__(self, file_list, scaler, seq_len=120, pred_len=5):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = scaler
        self.samples = []
        self._build(file_list)

    def _build(self, file_list):
        norm_indices = [i for i, c in enumerate(FEATURE_COLS) if c in NORM_COLS]
        for f in sorted(file_list):
            try:
                try:
                    df = pd.read_csv(f, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(f, encoding='gbk')
                    except UnicodeDecodeError:
                        df = pd.read_csv(f, encoding='utf-8')

                if '收盘价' not in df.columns or '日期' not in df.columns:
                    continue
                if len(df) < self.seq_len + self.pred_len + 30:
                    continue

                df['日期'] = pd.to_datetime(df['日期'])
                df = df.sort_values('日期').reset_index(drop=True)
                df = df.ffill().bfill().fillna(0)

                close = df['收盘价'].values
                eps = 1e-8

                # 均线乖离率
                for ma_name, ma_period in [('dev_ma5', 5), ('dev_ma10', 10),
                                           ('dev_ma20', 20), ('dev_ma30', 30),
                                           ('dev_ma60', 60)]:
                    ma = _sma(close, ma_period)
                    df[ma_name] = (close - ma) / (ma + eps) * 100

                df = add_daily_ta_features(df)

                for col in FEATURE_COLS:
                    if col not in df.columns:
                        df[col] = 0.0

                # 填补 TA 预热期的前导 NaN
                df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill().fillna(0)

                future_price = df['收盘价'].shift(-self.pred_len)
                df['Target_Return'] = (future_price - df['收盘价']) / (df['收盘价'] + eps) * 100
                df = df.dropna(subset=['Target_Return'])

                if len(df) < self.seq_len:
                    continue

                feat_raw = df[FEATURE_COLS].values.astype(np.float32)
                targets = df['Target_Return'].values

                # Meta
                log_mcap = np.log1p(float(df['总市值（元）'].iloc[-1])) if '总市值（元）' in df.columns else 0.0
                pe = np.clip(float(df['滚动市盈率'].iloc[-1]), 0, 200) / 50.0 if '滚动市盈率' in df.columns else 0.0
                pb = np.clip(float(df['市净率'].iloc[-1]), 0, 20) / 5.0 if '市净率' in df.columns else 0.0
                turnover_ma5 = float(np.nanmean(df['换手率'].values[-5:])) / 10.0 if '换手率' in df.columns else 0.0
                if '流通市值（元）' in df.columns and '总市值（元）' in df.columns:
                    circ_ratio = float(df['流通市值（元）'].iloc[-1]) / (float(df['总市值（元）'].iloc[-1]) + eps)
                else:
                    circ_ratio = 0.5
                ret_5d = float(df['涨幅%'].values[-5:].sum()) / 20.0 if '涨幅%' in df.columns else 0.0
                limit_up = float(df['是否涨停'].iloc[-1] == '是') if '是否涨停' in df.columns else 0.0
                meta_arr = np.array([log_mcap * 0.1, pe, pb, turnover_ma5, circ_ratio, ret_5d, limit_up], dtype=np.float32)

                for i in range(len(feat_raw) - self.seq_len + 1):
                    x = feat_raw[i:i + self.seq_len]
                    y = targets[i + self.seq_len - 1]
                    if np.isnan(x).any() or np.isinf(x).any() or np.isnan(y) or np.isinf(y):
                        continue
                    x_norm = x.copy()
                    x_norm[:, norm_indices] = self.scaler.transform(x_norm[:, norm_indices])
                    self.samples.append((x_norm.astype(np.float16), meta_arr.copy(), y))
            except Exception:
                continue

        print(f"[FileDataset] 构建完成: {len(self.samples)} 个样本, 来自 {len(file_list)} 个文件")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, meta, y = self.samples[idx]
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(meta), torch.tensor(y, dtype=torch.float32)

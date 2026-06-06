"""
EchoMarsh 流式股票数据集 (IterableDataset) — 解决全量数据内存爆炸
===============================================================
支持预计算缓存：第一次处理保存 .npz，后续直接读取，免去每轮重算 TA。

用法:
    ds = StreamStockDataset(
        data_dir="data/前复权",
        seq_len=120, pred_len=5,
        start_date='2010-01-01', end_date='2024-01-01',
        include_codes=('60', '00'),
        cache_dir="data/cache",     # 缓存目录，不设则不缓存
    )
    loader = DataLoader(ds, batch_size=256, num_workers=4)
"""

import os, glob, pickle, warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
try:
    import akshare as ak
except:
    ak = None  # 没有 akshare 也能跑，只是不加指数因子

FEATURE_COLS = [
    '开盘价', '最高价', '最低价', '收盘价', '成交量（股）', '成交额（元）',
    '涨幅%', '3日涨幅%', '10日涨幅%',
    '换手率', '振幅%', '量比',
    'dev_ma5', 'dev_ma10', 'dev_ma20', 'dev_ma30', 'dev_ma60',
    'rsi_14', 'macd', 'macd_diff', 'bb_pband', 'atr_14_norm',
    # 短线因子 (8个)
    'gap_open', 'upper_shadow_pct', 'lower_shadow_pct', 'body_pct',
    'limit_up_count_5d', 'vol_amt_ratio_20', 'price_position_20', 'return_entity',
    # 大盘语境 (2个)
    'csi300_ret', 'csi300_amt_ratio',
    # 高级因子 (4个)
    'hv_20', 'downside_vol_20', 'amihud_illiq', 'ret_skew_20',
]

NORM_COLS = [
    '开盘价', '最高价', '最低价', '收盘价',
    '成交量（股）', '成交额（元）',
    '换手率', '振幅%', 'atr_14_norm',
    'gap_open', 'body_pct', 'return_entity',
    'hv_20', 'downside_vol_20', 'amihud_illiq',
]


def _ema(x, period):
    return pd.Series(x).ewm(span=period, adjust=False, min_periods=1).mean().to_numpy()


def _sma(x, period):
    return pd.Series(x).rolling(window=period, min_periods=1).mean().to_numpy()


def add_ta_features(df):
    close = df['收盘价'].values.astype(np.float64)
    high = df['最高价'].values.astype(np.float64)
    low = df['最低价'].values.astype(np.float64)
    eps = 1e-10
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().to_numpy()
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().to_numpy()
    rs = avg_gain / (avg_loss + eps)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    df['macd'] = ema12 - ema26
    df['macd_diff'] = ema12 - ema26 - _ema(ema12 - ema26, 9)
    ma20 = _sma(close, 20)
    std20 = pd.Series(close).rolling(20, min_periods=1).std(ddof=0).to_numpy()
    df['bb_pband'] = (close - ma20) / (2 * std20 + eps) + 0.5
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    df['atr_14_norm'] = _ema(tr, 14) / (close + eps) * 100
    return df


def add_short_term_features(df):
    """新增短线因子：高开、影线、连板、放量等"""
    eps = 1e-10
    close = df['收盘价'].values
    open_ = df['开盘价'].values
    high = df['最高价'].values
    low = df['最低价'].values
    prev_close = df['前收盘价'].values if '前收盘价' in df.columns else close.copy()

    # 1. 高开幅度
    df['gap_open'] = (open_ / (prev_close + eps) - 1) * 100

    # 2. 上/下影线比例
    body_top = np.maximum(open_, close)
    body_bot = np.minimum(open_, close)
    candle_range = high - low + eps
    df['upper_shadow_pct'] = (high - body_top) / candle_range
    df['lower_shadow_pct'] = (body_bot - low) / candle_range

    # 3. 实体幅度 + 实体涨幅
    df['body_pct'] = np.abs(close - open_) / (prev_close + eps) * 100
    df['return_entity'] = (close - open_) / (prev_close + eps) * 100

    # 4. 过去5天涨停次数 (衰减加权)
    if '是否涨停' in df.columns:
        limit_up_flag = (df['是否涨停'] == '是').astype(float).values
    else:
        limit_up_flag = np.zeros(len(df))
    # 指数衰减权重: 最近的天权重最大
    limit_cnt = np.full_like(limit_up_flag, np.nan)
    for i in range(len(df)):
        if i < 5:
            limit_cnt[i] = limit_up_flag[:i+1].sum()
        else:
            weights = np.exp(np.arange(5) * 0.5)  # 越近越重
            limit_cnt[i] = np.average(limit_up_flag[i-4:i+1], weights=weights)
    df['limit_up_count_5d'] = limit_cnt

    # 5. 成交额相对20日均值
    if '成交额（元）' in df.columns:
        amt = df['成交额（元）'].values
        amt_ma20 = pd.Series(amt).rolling(20, min_periods=5).mean().to_numpy()
        df['vol_amt_ratio_20'] = amt / (amt_ma20 + eps)
    else:
        df['vol_amt_ratio_20'] = 1.0

    # 6. 在20日高低区间的位置
    roll_max = pd.Series(high).rolling(20, min_periods=5).max().to_numpy()
    roll_min = pd.Series(low).rolling(20, min_periods=5).min().to_numpy()
    df['price_position_20'] = (close - roll_min) / (roll_max - roll_min + eps)

    return df


def add_advanced_features(df):
    """高级因子：波动率、非流动性、偏度等量化研究公认因子"""
    eps = 1e-10
    ret = df['涨幅%'].values / 100.0
    amt = df['成交额（元）'].values

    # 1. 20日历史波动率（年化）
    df['hv_20'] = pd.Series(ret).rolling(20, min_periods=5).std().to_numpy() * np.sqrt(250)

    # 2. 20日下行波动率（只算负收益）
    neg_ret = ret.copy()
    neg_ret[neg_ret > 0] = 0
    df['downside_vol_20'] = pd.Series(neg_ret).rolling(20, min_periods=5).std().to_numpy() * np.sqrt(250)

    # 3. Amihud 非流动性指标 |ret| / 成交额(亿)
    #    值越大表示单位成交额对价格的冲击越大 → 流动性越差
    df['amihud_illiq'] = np.abs(ret) / (amt / 1e8 + eps)

    # 4. 20日收益率偏度
    df['ret_skew_20'] = pd.Series(ret).rolling(20, min_periods=5).skew().to_numpy()

    return df


# ─── 大盘指数数据缓存 ───
_CSI300_CACHE = None

def _load_csi300(cache_dir=None):
    """加载或下载沪深300日线，返回 DataFrame 或 None"""
    global _CSI300_CACHE
    if _CSI300_CACHE is not None:
        return _CSI300_CACHE
    cache_path = os.path.join(cache_dir or "/tmp", "csi300.pkl")
    if os.path.exists(cache_path):
        _CSI300_CACHE = pd.read_pickle(cache_path)
        return _CSI300_CACHE
    if ak is None:
        return None
    try:
        df = ak.stock_zh_index_daily(symbol="sh000300")
        df = df.rename(columns={'date': '日期', 'close': 'csi300_close', 'volume': 'csi300_vol'})
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)
        df['csi300_ret'] = df['csi300_close'].pct_change() * 100
        df['csi300_amt_ratio'] = df['csi300_vol'] / df['csi300_vol'].rolling(20).mean()
        _CSI300_CACHE = df[['日期', 'csi300_ret', 'csi300_amt_ratio']].dropna()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        _CSI300_CACHE.to_pickle(cache_path)
    except Exception as e:
        print(f"[warn] 沪深300加载失败: {e}")
        _CSI300_CACHE = pd.DataFrame()
    return _CSI300_CACHE


def process_csv(file_path, seq_len, pred_len, start_date, end_date, index_df=None):
    """读取 CSV，计算特征，返回 (feat_raw, targets, meta) 或 None

    targets: [5] = (1d_ret, 3d_ret, 5d_ret, 5d_max_ret, limit_flag)
    """
    try:
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='gbk')
    except:
        return None

    if '收盘价' not in df.columns or '日期' not in df.columns:
        return None
    if len(df) < seq_len + pred_len + 60:
        return None

    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期').reset_index(drop=True)
    if start_date:
        df = df[df['日期'] >= start_date]
    if end_date:
        df = df[df['日期'] <= end_date]
    if len(df) < seq_len + pred_len + 10:
        return None

    df = df.ffill().bfill().fillna(0)

    close = df['收盘价'].values
    eps = 1e-8
    for ma_name, ma_period in [('dev_ma5', 5), ('dev_ma10', 10), ('dev_ma20', 20), ('dev_ma30', 30), ('dev_ma60', 60)]:
        ma = _sma(close, ma_period)
        df[ma_name] = (close - ma) / (ma + eps) * 100

    df = add_ta_features(df)
    df = add_short_term_features(df)
    df = add_advanced_features(df)

    # 合并大盘指数因子
    if index_df is not None and len(index_df) > 0:
        idx = index_df[['日期', 'csi300_ret', 'csi300_amt_ratio']].copy()
        idx['日期'] = pd.to_datetime(idx['日期'])
        df = df.merge(idx, on='日期', how='left')
        df['csi300_ret'] = df['csi300_ret'].ffill().fillna(0)
        df['csi300_amt_ratio'] = df['csi300_amt_ratio'].ffill().fillna(1.0)
    else:
        df['csi300_ret'] = 0.0
        df['csi300_amt_ratio'] = 1.0

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill().fillna(0)

    # 多目标标签: [1d_ret, 3d_ret, 5d_ret, 5d_max_ret, limit_flag]
    close = df['收盘价'].values
    N = len(close)
    target_1d = np.full(N, np.nan)
    target_3d = np.full(N, np.nan)
    target_5d = np.full(N, np.nan)
    target_5d_max = np.full(N, np.nan)
    target_limit = np.zeros(N)

    for i in range(N):
        if i + 1 < N:
            target_1d[i] = (close[i+1] - close[i]) / close[i] * 100
        if i + 3 < N:
            target_3d[i] = (close[i+3] - close[i]) / close[i] * 100
        if i + pred_len < N:
            target_5d[i] = (close[i+pred_len] - close[i]) / close[i] * 100
            window_max = max(close[i:i+pred_len+1])
            target_5d_max[i] = (window_max - close[i]) / close[i] * 100
    if '是否涨停' in df.columns:
        for i in range(N):
            lookahead = df['是否涨停'].iloc[i+1:i+pred_len+1]
            if len(lookahead) > 0:
                target_limit[i] = float((lookahead == '是').any())

    # 构建5维标签矩阵
    label_arr = np.column_stack([target_1d, target_3d, target_5d, target_5d_max, target_limit])
    valid_rows = ~np.isnan(label_arr).any(axis=1)
    feat_raw = df[FEATURE_COLS].values.astype(np.float32)
    targets = label_arr
    valid = valid_rows & ~(np.isnan(feat_raw).any(axis=1) | np.isinf(feat_raw).any(axis=1))
    feat_raw = feat_raw[valid]
    targets = targets[valid]
    if len(feat_raw) < seq_len:
        return None

    log_mcap = np.log1p(float(df['总市值（元）'].iloc[-1])) if '总市值（元）' in df.columns else 0.0
    pe = np.clip(float(df['滚动市盈率'].iloc[-1]), 0, 200) / 50.0 if '滚动市盈率' in df.columns else 0.0
    pb = np.clip(float(df['市净率'].iloc[-1]), 0, 20) / 5.0 if '市净率' in df.columns else 0.0
    turnover_ma5 = float(np.nanmean(df['换手率'].values[-5:])) / 10.0 if '换手率' in df.columns else 0.0
    if '流通市值（元）' in df.columns and '总市值（元）' in df.columns:
        circ_ratio = float(df['流通市值（元）'].iloc[-1]) / (float(df['总市值（元）'].iloc[-1]) + eps)
    else:
        circ_ratio = 0.5
    ret_5d = float(df['涨幅%'].values[-5:].sum()) / 20.0
    limit_up = float(df['是否涨停'].iloc[-1] == '是') if '是否涨停' in df.columns else 0.0
    meta = np.array([log_mcap * 0.1, pe, pb, turnover_ma5, circ_ratio, ret_5d, limit_up], dtype=np.float32)

    return feat_raw, targets, meta


def build_samples(feat_raw, targets, meta, seq_len, scaler=None, norm_indices=None):
    """从特征矩阵中滑窗生成样本"""
    samples = []
    for i in range(len(feat_raw) - seq_len + 1):
        x = feat_raw[i:i + seq_len]
        y = targets[i + seq_len - 1]
        if np.isnan(x).any() or np.isinf(x).any() or np.isnan(y) or np.isinf(y):
            continue
        if scaler is not None and norm_indices is not None:
            x_norm = x.copy()
            x_norm[:, norm_indices] = scaler.transform(x_norm[:, norm_indices])
            samples.append((x_norm.astype(np.float16), meta, y))
        else:
            samples.append((x.astype(np.float32), meta, y))
    return samples


def get_cache_path(file_path, cache_dir, start_date, end_date):
    """获取缓存 .npz 路径（含日期范围，防止 train/val 覆盖）"""
    if not cache_dir:
        return None
    stem = os.path.splitext(os.path.basename(file_path))[0]
    date_suffix = ""
    if start_date:
        date_suffix += f"_{start_date.strftime('%Y%m%d') if hasattr(start_date, 'strftime') else str(start_date)[:10].replace('-','')}"
    if end_date:
        date_suffix += f"_{end_date.strftime('%Y%m%d') if hasattr(end_date, 'strftime') else str(end_date)[:10].replace('-','')}"
    return os.path.join(cache_dir, f"{stem}{date_suffix}.npz")


class StreamStockDataset(IterableDataset):
    """
    流式股票数据集 — 不把所有样本加载到内存。

    cache_dir: 如果设置，第一次处理后缓存 .npz，后续直接读取。
    """

    def __init__(self, data_dir, seq_len=120, pred_len=5,
                 start_date=None, end_date=None,
                 include_codes=None, max_files=None,
                 scaler=None, cache_dir=None):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.include_codes = include_codes
        self.max_files = max_files
        self.scaler = scaler
        self.cache_dir = cache_dir
        self.norm_indices = [i for i, c in enumerate(FEATURE_COLS) if c in NORM_COLS]

        self.files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if self.include_codes:
            basenames = [os.path.basename(f) for f in self.files]
            self.files = [f for f, bn in zip(self.files, basenames)
                          if any(bn.startswith(c) for c in self.include_codes)]
        if self.max_files:
            self.files = self.files[:self.max_files]

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            # 统计缓存命中
            cached = sum(1 for f in self.files if os.path.exists(get_cache_path(f, cache_dir, self.start_date, self.end_date)))
            if cached > 0:
                print(f"[StreamDataset] {len(self.files)} 个文件, "
                      f"缓存命中 {cached}/{len(self.files)}, "
                      f"scaler={'已提供' if scaler is not None else '未提供'}")
                return

        print(f"[StreamDataset] {len(self.files)} 个文件, "
              f"scaler={'已提供' if scaler is not None else '未提供'}")

        # 预加载大盘指数数据
        self.index_df = _load_csi300(self.cache_dir)

    def _load_file_data(self, file_path):
        """从缓存或 CSV 读取一个文件的数据，返回 (feat_raw, targets, meta) 或 None"""
        cache_path = get_cache_path(file_path, self.cache_dir, self.start_date, self.end_date)

        # 读缓存
        if cache_path and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                return data['feat_raw'], data['targets'], data['meta']
            except:
                pass  # 损坏的缓存，重新生成

        # 处理 CSV (传入大盘指数数据)
        result = process_csv(file_path, self.seq_len, self.pred_len,
                             self.start_date, self.end_date,
                             index_df=self.index_df)
        if result is None:
            return None

        feat_raw, targets, meta = result

        # 安全写入缓存（临时文件→重命名，防多 worker 冲突）
        if cache_path:
            tmp = cache_path + ".tmp"
            try:
                np.savez_compressed(tmp, feat_raw=feat_raw, targets=targets, meta=meta)
                if os.path.exists(tmp):
                    os.rename(tmp, cache_path)
            except:
                if os.path.exists(tmp):
                    os.remove(tmp)

        return feat_raw, targets, meta

    def fit_scaler(self, max_samples=500000):
        """遍历全量文件，采样拟合 StandardScaler"""
        print(f"[fit_scaler] 遍历 {len(self.files)} 个文件采样拟合 scaler...")
        buffer = []
        for idx, f in enumerate(self.files):
            data = self._load_file_data(f)
            if data is None:
                continue
            feat_raw, _, _ = data
            # 采样：均匀取最多 200 行
            step = max(1, len(feat_raw) // 200)
            for i in range(0, len(feat_raw), step):
                buffer.append(feat_raw[i])

            if len(buffer) >= max_samples:
                break
            if (idx + 1) % 500 == 0:
                print(f"  scaler 采样: {idx+1}/{len(self.files)} 文件, "
                      f"已收集 {len(buffer)} 行")

        concat = np.vstack(buffer)
        self.scaler = StandardScaler()
        self.scaler.fit(concat[:, self.norm_indices])
        print(f"[fit_scaler] 完成 (基于 {len(concat)} 行 {len(self.norm_indices)} 个特征)")
        return self.scaler

    def set_scaler(self, scaler):
        self.scaler = scaler

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.files
        if worker_info is not None:
            per_worker = len(files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(files)
            files = files[start:end]

        for f in files:
            data = self._load_file_data(f)
            if data is None:
                continue
            feat_raw, targets, meta = data
            samples = build_samples(feat_raw, targets, meta, self.seq_len,
                                    self.scaler, self.norm_indices)
            for x, meta_arr, y in samples:
                yield (
                    torch.from_numpy(x.astype(np.float32)),
                    torch.from_numpy(meta_arr),
                    torch.tensor(y, dtype=torch.float32),
                )

    def __len__(self):
        return len(self.files) * 2500


def create_stream_dataloaders(data_dir, seq_len=120, pred_len=5,
                               start_date='2010-01-01', end_date='2026-05-08',
                               include_codes=None, max_files=None,
                               batch_size=256, num_workers=8,
                               train_end_date='2024-01-01',
                               cache_dir=None):
    """创建 train/val 流式 DataLoader（共享 scaler）"""

    print("=== 第 1 步: 拟合 Scaler ===")
    scaler_ds = StreamStockDataset(
        data_dir, seq_len=seq_len, pred_len=pred_len,
        start_date=start_date, end_date=train_end_date,
        include_codes=include_codes, max_files=max_files,
        cache_dir=cache_dir,
    )
    scaler = scaler_ds.fit_scaler()
    del scaler_ds

    # 保存 Scaler（后续预测要用）
    scaler_path = os.path.join(os.path.dirname(data_dir), "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[Scaler] 已保存到 {scaler_path}")

    print("\n=== 第 2 步: 训练集 ===")
    train_ds = StreamStockDataset(
        data_dir, seq_len=seq_len, pred_len=pred_len,
        start_date=start_date, end_date=train_end_date,
        include_codes=include_codes, max_files=max_files,
        scaler=scaler, cache_dir=cache_dir,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True,
    )

    print("\n=== 第 3 步: 验证集 ===")
    val_ds = StreamStockDataset(
        data_dir, seq_len=seq_len, pred_len=pred_len,
        start_date=train_end_date, end_date=end_date,
        include_codes=include_codes, max_files=max_files,
        scaler=scaler, cache_dir=cache_dir,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader

"""
EchoMarsh (鸣泽) 每日市场扫描与预测更新
========================================
收市后一键运行，完成：
  1. 拉取全市场行情与外围指数
  2. 获取涨停板候选池
  3. 多因子评分 + 模型预测（如模型可用）
  4. 与上一交易日对比，标记分数变化
  5. 输出推荐清单

用法:
    python scripts/daily_update.py
    python scripts/daily_update.py --date 20260513
"""

import os
import sys
import json
import argparse
import traceback
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
import pandas as pd
os.environ["NO_PROXY"] = "*"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
import akshare as ak

from core.scanner.persistence import PersistenceManager
from core.scanner.daily_scanner import DailyScannerEngine
from core.scanner.global_market import GlobalMarketEngine

# ──────────────────────────────────────────
# 特征配置（与训练时 offline_dataset.py 一致）
# ──────────────────────────────────────────
FEATURE_COLS = [
    # 基础量价 (6)
    '开盘价', '最高价', '最低价', '收盘价', '成交量（股）', '成交额（元）',
    # 衍生收益率 (3)
    '涨幅%', '3日涨幅%', '10日涨幅%',
    # 流动性与波动 (3)
    '换手率', '振幅%', '量比',
    # 均线偏离度 % (5)
    'ma5_dev%', 'ma10_dev%', 'ma20_dev%', 'ma60_dev%', 'ma120_dev%',
    # 技术指标 (5)
    'rsi_14', 'macd', 'macd_diff', 'bb_pctb', 'atr_14_norm',
]

NORM_COLS = ['开盘价','最高价','最低价','收盘价','成交量（股）','成交额（元）','换手率','振幅%','atr_14_norm']

SEQ_LEN = 120
PRED_LEN = 5


# ──────────────────────────────────────────
# 特征工程（纯 numpy/pandas，与训练保持一致）
# ──────────────────────────────────────────

def _ema(arr, span):
    """pandas ewm 实现指数移动平均"""
    s = pd.Series(arr)
    return s.ewm(span=span, min_periods=1, adjust=False).mean().values

def _sma(arr, window):
    """简单移动平均"""
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=1).mean().values

def add_daily_ta_features(df):
    """计算技术指标，原地修改 df"""
    close = df['收盘价'].values.astype(float)
    high = df['最高价'].values.astype(float)
    low = df['最低价'].values.astype(float)
    vol = df['成交量（股）'].values.astype(float)
    eps = 1e-10

    # RSI 14 (Wilder 平滑)
    diff = np.diff(close, prepend=close[0])
    gain = np.where(diff > 0, diff, 0)
    loss = np.where(diff < 0, -diff, 0)
    avg_gain = _ema(gain, 14)
    avg_loss = _ema(loss, 14)
    rs = avg_gain / (avg_loss + eps)
    df['rsi_14'] = 100 - 100 / (1 + rs)

    # MACD (12, 26, 9)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    df['macd'] = macd
    df['macd_diff'] = macd - signal

    # 布林带 %b (20, 2)
    ma20 = _sma(close, 20)
    std20 = pd.Series(close).rolling(window=20, min_periods=1).std().values
    df['bb_pctb'] = (close - ma20 + 2 * std20) / (4 * std20 + eps)

    # ATR14 (归一化到收盘价 %)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr14 = _ema(tr, 14)
    df['atr_14_norm'] = atr14 / (close + eps) * 100

    # 涨幅
    df['涨幅%'] = np.r_[0, np.diff(close) / close[:-1]] * 100
    df['3日涨幅%'] = np.r_[0, 0, 0, (close[3:] - close[:-3]) / close[:-3] * 100][:len(close)]
    df['10日涨幅%'] = np.r_[[0]*10, (close[10:] - close[:-10]) / close[:-10] * 100][:len(close)]

    # 量比 (当日量 / 5日均量)
    vol_ma5 = _sma(vol, 5)
    df['量比'] = vol / (vol_ma5 + eps)

    # 均线偏离度
    for ma in [5, 10, 20, 60, 120]:
        ma_vals = _sma(close, ma)
        df[f'ma{ma}_dev%'] = (close - ma_vals) / (ma_vals + eps) * 100

    return df


def compute_meta_features(df):
    """
    从 dataframe 最后一行提取 7 维 meta 特征。
    返回 dict 或 None（数据不足时）
    """
    if len(df) < 60:
        return None
    last = df.iloc[-1]
    eps = 1e-10

    # 对数市值 × 0.1 (缩放到 ~1-10 量级)
    liutong = float(last.get('流通市值', 0)) or float(last.get('总市值', 0)) or 1e8
    log_mcap = np.log(max(liutong, 1e8)) * 0.1

    # PE / 50, PB / 5
    pe = float(last.get('市盈率-动态', 0)) or 0
    pb = float(last.get('市净率', 0)) or 0

    # 换手率 MA5 / MA10
    turnover = df['换手率'].values.astype(float)
    turn_ma5 = np.mean(turnover[-5:]) if len(turnover) >= 5 else 0
    turn_ma10 = np.mean(turnover[-10:]) if len(turnover) >= 10 else 0

    # 流通比例
    total_mv = float(last.get('总市值', 0)) or 1
    circ_ratio = liutong / total_mv if total_mv > 0 else 0.5

    # 近 5 日涨幅和 / 20
    close = df['收盘价'].values.astype(float)
    ret_5d = (close[-1] - close[-min(6, len(close))]) / (close[-min(6, len(close))] + eps) * 100
    ret_5d_norm = max(min(ret_5d / 20, 5), -5)

    # 涨停标记 (当日涨幅 > 9.5%)
    limit_up = 1.0 if abs(float(last.get('涨幅%', 0))) > 9.5 else 0.0

    return np.array([
        log_mcap,
        min(pe / 50, 5) if pe > 0 else 0,
        min(pb / 5, 2) if pb > 0 else 0,
        min(turn_ma5 / 10, 5),
        min(turn_ma10 / 10, 5),
        circ_ratio,
        ret_5d_norm,
    ], dtype=np.float32)


def compute_features_for_stock(code: str, end_date: str) -> tuple:
    """
    获取一只股票的 120 天历史，计算 22 维特征 + 7 维 meta。
    返回 (ts_feats, meta_arr, name, industry) 或 (None, None, None, None)
    """
    try:
        start_dt = (pd.Timestamp(end_date) - pd.Timedelta(days=300)).strftime('%Y%m%d')
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start_dt, end_date=end_date,
                                adjust="qfq")
        if df is None or len(df) < SEQ_LEN + 30:
            return None, None, None, None

        # 统一列名
        df.rename(columns={
            '开盘': '开盘价', '收盘': '收盘价', '最高': '最高价', '最低': '最低价',
            '成交量': '成交量（股）', '成交额': '成交额（元）',
            '换手率': '换手率', '振幅': '振幅%',
        }, inplace=True)

        # 补齐缺失列
        for col in ['开盘价', '最高价', '最低价', '收盘价', '成交量（股）', '成交额（元）', '换手率', '振幅%']:
            if col not in df.columns:
                return None, None, None, None

        name = str(df.iloc[-1].get('名称', code))
        industry = str(df.iloc[-1].get('行业', '未知'))

        # 计算技术指标
        add_daily_ta_features(df)

        # 补齐缺失列
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        # 取最后 SEQ_LEN 行
        feats = df[FEATURE_COLS].values[-SEQ_LEN:].astype(np.float32)

        # Meta
        meta = compute_meta_features(df)

        if len(feats) < SEQ_LEN or meta is None:
            return None, None, None, None

        return feats, meta, name, industry

    except Exception as e:
        print(f"    [跳过] {code} 特征计算失败: {e}")
        return None, None, None, None


# ──────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────

def run_daily_update(date_str: str = None):
    if date_str is None:
        date_str = datetime.now().strftime('%Y%m%d')

    print("=" * 60)
    print(f"  EchoMarsh 每日市场更新 | {date_str}")
    print("=" * 60)

    # 初始化
    db = PersistenceManager()
    scanner = DailyScannerEngine(board_filter='all')

    # 外围指数
    print("\n[1/5] 外围指数...")
    try:
        gm = GlobalMarketEngine()
        indices = gm.fetch_global_indices()
        for idx in indices:
            print(f"  {idx.get('name','?')}: {idx.get('close',0):.2f} ({idx.get('pct_change',0):+.2f}%)")
    except Exception as e:
        print(f"  [跳过] 外围指数: {e}")

    # 获取候选池（涨停板 + 近期强势股）
    print("\n[2/5] 候选池...")
    candidates = scanner._get_candidates(date_str)
    print(f"  涨停候选: {len(candidates)} 只")

    # 尝试加载模型（可选）
    model = None
    device = None
    checkpoint_path = os.path.join(project_root, "models", "checkpoints", "best_echomarsh_model.pth")
    scaler_path = os.path.join(project_root, "models", "checkpoints", "scaler.npy")
    scaler_mean = None
    scaler_std = None

    if os.path.exists(checkpoint_path):
        try:
            print("\n[3/5] 加载模型...")
            import torch
            from models.backbone.model_factory import ModelFactory
            model, device = ModelFactory.create_model(model_type='transformer', ts_feature_dim=36, meta_feature_dim=7)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            model.eval()
            print(f"  模型已加载 ({device})")

            if os.path.exists(scaler_path):
                scaler_data = np.load(scaler_path, allow_pickle=True).item()
                scaler_mean = scaler_data['mean']
                scaler_std = scaler_data['std']
                print(f"  Scaler 已加载")
            else:
                print(f"  [警告] scaler.npy 未找到，跳过归一化")
        except Exception as e:
            print(f"  [跳过] 模型加载失败: {e}")
            model = None
    else:
        print("\n[3/5] 模型未训练，使用多因子评分模式")

    # 逐票评分
    print(f"\n[4/5] 逐票评分 ({len(candidates)} 只)...")
    predictions = []
    total = len(candidates)

    for i, cand in enumerate(candidates):
        code = cand['code']
        name = cand['name']
        print(f"  ({i+1}/{total}) {code} {name}...", end=' ')

        try:
            # 多因子评分
            sector_ranks = scanner._get_sector_ranks(date_str)
            stock = scanner._score_stock(cand, sector_ranks, date_str)

            model_ret_pred = None
            model_limit_up_prob = None

            # 模型预测
            if model is not None:
                feats, meta_arr, _, _ = compute_features_for_stock(code, date_str)
                if feats is not None:
                    import torch
                    # 归一化（仅 NORM_COLS）
                    if scaler_mean is not None and scaler_std is not None:
                        for j, col in enumerate(FEATURE_COLS):
                            if col in NORM_COLS:
                                feats[:, j] = (feats[:, j] - scaler_mean[j]) / (scaler_std[j] + 1e-10)

                    with torch.no_grad():
                        ts_tensor = torch.from_numpy(feats).unsqueeze(0).to(device)
                        meta_tensor = torch.from_numpy(meta_arr).unsqueeze(0).to(device)
                        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                            output = model(ts_tensor, meta_tensor)
                        model_ret_pred = float(output[0, 0].cpu())
                        model_limit_up_prob = float(torch.sigmoid(output[0, 4]).cpu())

            # 综合评分: 因子分(0-100) + 模型分归一化(0-100)
            factor_score = stock.total_score
            if model_ret_pred is not None and model_limit_up_prob is not None:
                model_score = min(max(model_ret_pred * 5 + model_limit_up_prob * 30, 0), 100)
                combined = 0.5 * factor_score + 0.5 * model_score
            else:
                model_score = None
                combined = factor_score

            predictions.append({
                'code': code,
                'name': name,
                'industry': stock.industry,
                'factor_score': factor_score,
                'model_ret_pred': model_ret_pred,
                'model_limit_up_prob': model_limit_up_prob,
                'combined_score': round(combined, 1),
                'lianban': stock.lianban,
            })

            print(f"→ {combined:.0f}分")
            if model_ret_pred is not None:
                print(f"     模型预测: 收益={model_ret_pred:+.2f}% 涨停概率={model_limit_up_prob:.1%}")

        except Exception as e:
            print(f"→ 失败: {e}")
            traceback.print_exc()

    # 排序
    predictions.sort(key=lambda x: x['combined_score'], reverse=True)

    # 存储
    print(f"\n[5/5] 保存结果...")
    db.save_predictions(date_str, predictions)

    # 对比上一交易日
    dates = db.get_prediction_dates(2)
    prev_date = dates[1] if len(dates) >= 2 else None
    if prev_date:
        changes = db.get_prediction_with_changes(date_str, prev_date)
        print(f"\n{'='*60}")
        print(f"对比 {prev_date} 分数变化:")
        print(f"{'代码':<8} {'名称':<8} {'综合分':<8} {'前日':<8} {'变化':<8} {'连板'}")
        print("-" * 60)
        for r in changes[:20]:
            chg = r['change']
            chg_str = f"{chg:+.1f}" if chg is not None else "N/A"
            print(f"{r['code']:<8} {r['name']:<8} {r['combined_score']:<8.1f} "
                  f"{r.get('prev_score',0):<8.1f} {chg_str:<8} {r.get('lianban',0)}")
    else:
        print(f"\n  首次运行，无历史对比数据")

    print(f"\n{'='*60}")
    print(f"  更新完成！{len(predictions)} 只股票已评分")
    print(f"  查看结果: streamlit run scripts/web_app.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoMarsh 每日更新")
    parser.add_argument("--date", type=str, default=None, help="日期 YYYYMMDD，默认今天")
    args = parser.parse_args()
    run_daily_update(args.date)

"""
EchoMarsh (鸣泽) 全局配置中心
============================
所有超参数和路径集中管理，各模块通过 `from config import ...` 引用，
避免硬编码散布在多个脚本中。
"""

import os

# ============================================================
# 项目路径（自动检测，无需修改）
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "stocks")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# ============================================================
# 模型超参数
# ============================================================
class ModelConfig:
    """CNN-Transformer 架构超参数"""
    model_type: str = "transformer"
    ts_feature_dim: int = 32     # 日线特征维度（对齐 FEATURE_COLS）
    meta_feature_dim: int = 7    # 元特征维度（市值/PE/PB/换手率等）
    d_model: int = 256           # Transformer 隐藏维度
    nhead: int = 8               # 多头注意力头数
    num_layers: int = 4          # Transformer 编码器层数
    dim_feedforward: int = 512   # 前馈网络维度
    dropout: float = 0.1         # Dropout 比例

# ============================================================
# 训练超参数
# ============================================================
class TrainConfig:
    """训练超参数（可按场景覆写）"""
    seq_len: int = 120           # 回溯窗口（天）
    pred_len: int = 5            # 预测窗口（天）
    batch_size: int = 256        # 批次大小（4090 24GB 可用 256-512）
    num_workers: int = 4         # DataLoader 工作进程
    epochs: int = 200            # 最大训练轮数
    lr: float = 1e-4             # 初始学习率
    patience: int = 20           # 早停耐心值（验证 loss 不降则停）
    use_amp: bool = True         # 混合精度训练
    grad_clip: float = 1.0       # 梯度裁剪阈值

# ============================================================
# 数据过滤
# ============================================================
class DataConfig:
    """数据与股票筛选"""
    include_codes: tuple = ('60', '00')   # 主板代码前缀
    train_start: str = "2010-01-01"
    train_end: str = "2024-01-01"
    val_start: str = "2024-01-01"
    val_end: str = "2026-05-08"

# ============================================================
# 风控参数
# ============================================================
class RiskConfig:
    """交易风控约束"""
    max_positions: int = 10      # 最大持仓数
    max_daily_orders: int = 10   # 每日最大下单次数
    stop_loss_pct: float = -5.0  # 止损线（%）
    take_profit_pct: float = 8.0 # 止盈线（%）
    min_confidence: float = 0.55 # 最低模型置信度阈值
    cash_reserve_pct: float = 0.05  # 最低现金保留比例
    single_position_max_pct: float = 0.20  # 单票最大仓位

# ============================================================
# 因子评分配置
# ============================================================
class FactorConfig:
    """多因子评分权重（总分 100）"""
    consecutive_limit_up: int = 15     # 连板强度
    seal_quality: int = 10             # 封板质量
    main_capital: int = 12             # 主力资金
    tech_momentum: int = 8             # 技术动量
    tech_trend: int = 7                # 技术趋势
    sector_resonance: int = 10         # 板块共振
    community_sentiment: int = 10      # 社区情绪
    lhb_seat: int = 8                  # 龙虎榜席位
    turnover_anomaly: int = 5          # 换手率异动
    sentiment_flow_divergence: int = 5 # 情绪资金背离
    market_cap_fit: int = 5            # 市值适配
    safety_filter: int = 5             # 安全过滤

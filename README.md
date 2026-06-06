# EchoMarsh (鸣泽) — A 股量化选股框架

基于 **CNN-Transformer 混合架构**的深度学习选股系统，覆盖数据采集 → 特征工程 → 模型训练 → 多因子评分 → 回测评估 → 交易执行的完整链路。

## 技术栈

- **模型**: 1D-CNN 局部特征提取 + Transformer 多头自注意力 + 多头预测（收益率 / 涨停概率 / 方向分类），~7M 参数
- **训练**: Focal Loss（涨停样本 ~1% 极度不平衡）、Huber Loss、AMP 混合精度、CosineAnnealingWarmRestarts、早停
- **特征工程**: 32 维日线因子（RSI/MACD/Bollinger/ATR/Amihud 非流动性/下行波动率/收益偏度/蜡烛图形态）+ 7 维元特征（市值/PE/PB/换手率/流通比/5日动量/涨停标记）
- **评估体系**: 横截面 IC / Rank IC / Information Ratio / 分层回测 / Walk-Forward 滚动验证
- **数据源**: akshare（东方财富 / 财联社），curl_cffi 绕过 TLS 指纹检测

## 项目结构

```
EchoMarsh/
├── config.py                       # 全局配置中心（模型/训练/数据/风控/因子权重）
├── requirements.txt                # 依赖清单
├── core/
│   ├── scanner/                    # 市场扫描 & 多因子评分
│   │   ├── daily_scanner.py        # 12 因子 A 股短线选股引擎
│   │   ├── data_fetcher.py         # 竞价量比 / 首板数据采集
│   │   ├── community_sentiment.py  # 东方财富热度榜 / 飙升榜
│   │   ├── market_sentiment.py     # 市场整体情绪（涨停数/炸板率/跌停数）
│   │   ├── global_market.py        # 美股/A50/港股外围 + 新闻 + 持仓顾问
│   │   ├── fundamental_fetcher.py  # 基本面（市值/PE/龙虎榜席位）
│   │   ├── sector_fetcher.py       # 行业板块与个股偏离度
│   │   ├── nlp_processor.py        # 文本情绪分析
│   │   └── persistence.py          # SQLite 持久化（5 张表完整 CRUD）
│   ├── trainer/                    # 训练引擎
│   │   ├── trainer.py              # Focal Loss + Huber + AMP + 早停
│   │   ├── offline_dataset.py      # 日线数据集（32 维特征 + 5 维标签）
│   │   └── stream_dataset.py       # 流式 IterableDataset（全市场 5839 只股票）
│   ├── evaluation/                 # 因子评估
│   │   └── factor_eval.py          # IC / Rank IC / IR / 分层回测
│   └── executor/
│       └── qmt_executor.py         # QMT 交易执行（实盘/模拟双模式 + 风控）
├── models/backbone/
│   ├── transformer_net.py          # CNN → PositionalEncoding → Transformer → Fusion
│   └── model_factory.py            # 模型工厂 + Xavier 初始化
└── scripts/
    ├── train.py / train_v2.py      # 离线训练
    ├── train_stream.py             # 流式训练
    ├── walk_forward_train.py       # 滚动前向验证（6 窗口）
    ├── predict.py                  # 批量 GPU 推理
    ├── daily_update.py             # 每日扫描 → 评分 → 存储 → 对比
    ├── daily_report.py             # 终端 + HTML 日报
    ├── daily_review.py             # 复盘分析 + 在线微调
    ├── paper_trade.py              # 回测（总收益/Sharpe/MaxDD/胜率）
    ├── dashboard.py                # Rich 终端仪表盘
    └── web_app.py                  # Streamlit Web 界面
```

## 因子体系

### 训练特征（32 维日线）

| 类别 | 因子 | 说明 |
|------|------|------|
| 价量基础 | 开高低收 / 成交量 / 成交额 / 换手率 / 振幅 / 量比 | 标准化日线行情 |
| 趋势指标 | RSI-14 / MACD / Bollinger %b / ATR-14 / MA 乖离率(5/10/20/30/60) | 多周期技术指标 |
| 短线形态 | 高开幅度 / 上影线/下影线比例 / 实体幅度 / 衰减加权涨停计数 | 超短线交易特征 |
| 高级因子 | 历史波动率 / 下行波动率 / Amihud 非流动性 / 收益偏度 | 量化研究公认因子 |
| 市场语境 | 沪深 300 收益率 / 成交额比 | 大盘联动特征 |

### 12 因子评分体系（100 分制）

| # | 因子 | 权重 | 逻辑 |
|---|------|:---:|------|
| 1 | 连板强度 | 15 | 连续涨停天数 × 衰减系数 |
| 2 | 封板质量 | 10 | 封单量 / 流通市值 |
| 3 | 主力资金 | 12 | 主力净流入占比 |
| 4 | 技术动量 | 8 | RSI + MACD + 量比综合 |
| 5 | 技术趋势 | 7 | 多周期均线多头排列 |
| 6 | 板块共振 | 10 | 所属板块强度排名 |
| 7 | 社区情绪 | 10 | 东财热度榜 + 飙升榜 |
| 8 | 龙虎榜席位 | 8 | 拉萨天团检测（越低越好） |
| 9 | 换手率异动 | 5 | 放量滞涨 / 缩量加速 |
| 10 | 情绪资金背离 | 5 | 杀猪盘 / 诱多出货检测 |
| 11 | 市值适配 | 5 | 流通市值分档适配 |
| 12 | 安全过滤 | 5 | ST / 退市 / 财务暴雷 |

## 风控体系

- 单票最大仓位 20%，最低现金保留 5%
- 止损 -5% / 止盈 +8% / 移动止盈（跌破 5 日线减仓）
- 每日最大下单 10 次，最低置信度 55%
- T+1 合规 / 做 T 高抛策略（+3% 卖 1/3）
- 多窗口策略调度（竞价 → 早盘 → 日内 → 尾盘 → 次日开盘）

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 准备数据：将前复权日线 CSV 放入 data/stocks/（文件名为股票代码如 600001.csv）

# 训练
python scripts/train.py            # 离线训练
python scripts/train_stream.py     # 流式训练（全市场）

# 预测
python scripts/predict.py          # 批量推理，输出 Top 20

# 每日扫描
python scripts/daily_update.py

# 因子评估
python core/evaluation/factor_eval.py

# 回测
python scripts/paper_trade.py

# Web 看板
streamlit run scripts/web_app.py
```

## 评估指标说明

| 指标 | 含义 | 优秀阈值 |
|------|------|:---:|
| IC Mean | 预测值与实际收益的 Pearson 相关系数 | > 0.03 |
| Rank IC | Spearman 秩相关系数（更稳健） | > 0.05 |
| IR | IC Mean / IC Std（信息比率） | > 0.5 |
| IC > 0 占比 | IC 为正的交易日比例 | > 55% |
| Long-Short | Top 10% 减 Bottom 10% 收益差 | 显著为正 |

## 版本

- **v0.01**: 初始版本 — 量化选股与训练框架
- **v0.02**: 新增 Stream 数据集、predict/train_v2/train_stream 脚本
- **v0.02b**: 修复 lookahead bias / 维度不一致 / API 崩溃 / 空壳模块
- **v0.03**: 新增 requirements.txt / 统一配置中心 / 因子 IC 评估模块 / logging / 专业 README

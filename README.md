# EchoMarsh (鸣泽) — 量化选股与训练框架

A股量化选股框架，基于 CNN-Transformer 混合架构的深度学习模型，支持全市场扫描、多因子评分、滚动训练和涨停预测。

## 项目结构

```
EchoMarsh/
├── config.py                    # 全局配置
├── main.py                      # 入口
├── core/
│   ├── scanner/                 # 市场扫描 & 数据采集
│   │   ├── daily_scanner.py     # 每日全市场扫描
│   │   ├── data_fetcher.py      # 行情数据拉取
│   │   ├── dataset.py           # 数据集构建
│   │   ├── preprocessor.py      # 数据预处理
│   │   ├── community_sentiment.py # 社区情绪分析
│   │   ├── market_sentiment.py  # 市场情绪
│   │   ├── global_market.py     # 外围指数
│   │   ├── fundamental_fetcher.py # 基本面数据
│   │   ├── sector_fetcher.py    # 行业板块
│   │   ├── nlp_processor.py     # NLP 文本处理
│   │   ├── multimodal_preprocessor.py # 多模态预处理
│   │   ├── persistence.py       # 数据持久化
│   │   └── mock_data_gen.py     # 模拟数据生成
│   ├── trainer/                 # 训练模块
│   │   ├── trainer.py           # 训练中枢
│   │   ├── offline_dataset.py   # 离线全量数据集
│   │   └── stream_dataset.py    # 流式 IterableDataset
│   └── executor/                # 交易执行
│       └── qmt_executor.py      # QMT 执行器
├── models/
│   └── backbone/
│       ├── transformer_net.py   # CNN-Transformer 网络
│       └── model_factory.py     # 模型工厂
└── scripts/
    ├── train.py                 # 训练脚本 (离线版)
    ├── train_v2.py              # 训练脚本 v0.02
    ├── train_stream.py          # 流式训练 (全市场)
    ├── walk_forward_train.py    # 滚动前向验证
    ├── predict.py               # 批量预测
    ├── daily_update.py          # 每日扫描更新
    ├── daily_report.py          # 日报生成
    ├── daily_review.py          # 复盘分析
    ├── dashboard.py             # 仪表盘
    ├── paper_trade.py           # 模拟交易
    └── web_app.py               # Web 界面
```

## 模型架构

- **CNN 前置**: 1D 卷积提取时序局部特征
- **Transformer 主体**: 多头自注意力捕捉长程依赖
- **多头输出**: 预测未来 N 日收益率 + 涨停概率 + 涨跌分类
- **参数规模**: ~7M (d_model=256, 4层, 8头)

## 快速开始

### 安装依赖

```bash
pip install torch numpy pandas scikit-learn akshare
```

### 准备数据

将前复权日线 CSV 放入 `data/stocks/` 目录。每个文件以股票代码命名，如 `600001.csv`。

### 训练

```bash
python scripts/train.py          # 离线全量训练
python scripts/train_v2.py       # v0.02 改进版
python scripts/train_stream.py   # 流式训练（适合内存受限环境）
```

### 预测

```bash
python scripts/predict.py        # 全市场批量推理，输出 Top 推荐
```

### 每日扫描

```bash
python scripts/daily_update.py
```

## 配置

在对应脚本中修改超参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SEQ_LEN` | 120 | 回溯窗口（天）|
| `PRED_LEN` | 5 | 预测窗口（天）|
| `BATCH_SIZE` | 256 | 批次大小 |
| `EPOCHS` | 200 | 训练轮数 |
| `LR` | 1e-4 | 学习率 |
| `PATIENCE` | 20 | 早停耐心值 |

## 版本

- **v0.01**: 初始版本 — 量化选股与训练框架
- **v0.02**: 新增 Stream 数据集、predict/train_v2/train_stream 脚本

"""
EchoMarsh v0.02 训练脚本 — 简单可靠，每轮保存
=============================================
用法:
    python scripts/train_v2.py
"""
import os, sys, pickle, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.backbone.model_factory import ModelFactory
from core.trainer.trainer import EchoMarshTrainer
from core.trainer.offline_dataset import OfflineStockDataset

# ========== 配置 ==========
DATA_DIR = os.path.join(project_root, "data", "stocks")
SAVE_DIR = os.path.join(project_root, "models", "checkpoints_v2")
SEQ_LEN, PRED_LEN = 120, 5
BATCH_SIZE = 256
EPOCHS = 200
PATIENCE = 20
LR = 1e-4
NUM_WORKERS = 4
INCLUDE_CODES = ('60', '00')  # 主板

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"模型保存到: {SAVE_DIR}")

# ========== 第1步: 构建数据集（全量主板） ==========
print(f"\n=== 第1步: 加载数据 ===")
print(f"数据路径: {DATA_DIR}")
print(f"筛选: 代码前缀 {INCLUDE_CODES}")

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
if INCLUDE_CODES:
    files = [f for f in files if os.path.basename(f).startswith(INCLUDE_CODES)]
print(f"主板股票: {len(files)} 只")

# 构建数据集
train_ds = OfflineStockDataset(
    DATA_DIR, seq_len=SEQ_LEN, pred_len=PRED_LEN,
    is_train=True, start_date='2010-01-01', end_date='2024-01-01',
    include_codes=INCLUDE_CODES,
)
train_scaler = train_ds.scaler

# 保存 scaler
scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(train_scaler, f)
print(f"[Scaler] 已保存: {scaler_path}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True)
del train_ds

val_ds = OfflineStockDataset(
    DATA_DIR, seq_len=SEQ_LEN, pred_len=PRED_LEN,
    is_train=False, start_date='2024-01-01', end_date='2026-05-14',
    include_codes=INCLUDE_CODES,
)
val_ds.set_scaler(train_scaler)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True)
del val_ds

print(f"训练集: {len(train_loader)} batch | 验证集: {len(val_loader)} batch")

# ========== 第2步: 训练 ==========
print(f"\n=== 第2步: 训练 ===")
model, device = ModelFactory.create_model('transformer', ts_feature_dim=32, meta_feature_dim=7)
print(f"设备: {device}")

trainer = EchoMarshTrainer(
    model=model, device=device,
    checkpoint_dir=SAVE_DIR,
    lr=LR, epochs=EPOCHS, patience=PATIENCE, use_amp=True,
)
trainer.fit(train_loader, val_loader)

print(f"\n训练完成！最佳模型: {os.path.join(SAVE_DIR, 'best_echomarsh_model.pth')}")
print(f"Scaler: {scaler_path}")

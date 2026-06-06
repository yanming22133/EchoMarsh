"""
EchoMarsh (鸣泽) 量化系统 - 训练中枢 (日线版)
============================================
用法:
    python scripts/train.py
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from torch.utils.data import DataLoader
from core.trainer.offline_dataset import OfflineStockDataset
from models.backbone.model_factory import ModelFactory
from core.trainer.trainer import EchoMarshTrainer


def main():
    print("=" * 60)
    print("      EchoMarsh (鸣泽) 量化系统 - 训练中枢")
    print("=" * 60)

    # 数据路径
    data_dir = os.path.join(project_root, "data", "stocks")
    checkpoint_dir = os.path.join(project_root, "models", "checkpoints")

    # 超参数
    SEQ_LEN = 120
    PRED_LEN = 5
    BATCH_SIZE = 256          # 4090 24GB
    NUM_WORKERS = 8           # AutoDL 多核
    EPOCHS = 200
    LR = 1e-4
    PATIENCE = 20

    # 主板筛选
    INCLUDE_CODES = ('60', '00')

    print(f"[配置] seq_len={SEQ_LEN}, pred_len={PRED_LEN}, "
          f"batch={BATCH_SIZE}, workers={NUM_WORKERS}")

    # 训练集: 2010-01 ~ 2024-01
    print(f"\n加载训练数据: {data_dir}")
    train_dataset = OfflineStockDataset(
        data_dir, seq_len=SEQ_LEN, pred_len=PRED_LEN,
        is_train=True, start_date='2010-01-01', end_date='2024-01-01',
        include_codes=INCLUDE_CODES,
    )
    train_scaler = train_dataset.scaler
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    del train_dataset

    # 验证集: 2024-01 ~ 2026-05
    print(f"\n加载验证数据...")
    val_dataset = OfflineStockDataset(
        data_dir, seq_len=SEQ_LEN, pred_len=PRED_LEN,
        is_train=False, start_date='2024-01-01', end_date='2026-05-08',
        include_codes=INCLUDE_CODES,
    )
    val_dataset.set_scaler(train_scaler)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    del val_dataset

    if len(train_loader) == 0:
        print("[ERROR] 训练集为空，请检查数据路径。")
        return

    # 初始化模型
    model, device = ModelFactory.create_model(
        model_type='transformer',
        ts_feature_dim=36,
        meta_feature_dim=7,
    )

    # 训练
    trainer = EchoMarshTrainer(
        model=model,
        device=device,
        checkpoint_dir=checkpoint_dir,
        lr=LR,
        epochs=EPOCHS,
        patience=PATIENCE,
        use_amp=True,
    )

    trainer.fit(train_loader, val_loader)

    print("\n训练完成！最佳模型保存在:", os.path.join(checkpoint_dir, "best_echomarsh_model.pth"))


if __name__ == "__main__":
    main()

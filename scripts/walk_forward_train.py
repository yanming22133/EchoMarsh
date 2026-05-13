"""
EchoMarsh (鸣泽) 量化系统 - 滚动前向验证 (Walk-Forward)
=====================================================
逐步滚动训练窗口，模拟实盘场景下的模型退化检测。

用法:
    python scripts/walk_forward_train.py
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from core.trainer.offline_dataset import OfflineStockDataset
from models.backbone.model_factory import ModelFactory
from core.trainer.trainer import EchoMarshTrainer


def run_walk_forward_validation():
    print("=" * 60)
    print("   EchoMarsh 滚动前向验证 (Walk-Forward)")
    print("=" * 60)

    data_dir = r"f:\EchoMarsh\5.8号\每只股票一个文件\前复权\前复权"
    checkpoint_dir = os.path.join(project_root, "models", "checkpoints", "walk_forward")
    os.makedirs(checkpoint_dir, exist_ok=True)

    SEQ_LEN = 120
    PRED_LEN = 5
    BATCH_SIZE = 32
    NUM_WORKERS = 2

    # 滚动窗口定义 (train_start, train_end, val_end)
    windows = [
        ("2018-01-01", "2020-01-01", "2021-01-01"),
        ("2019-01-01", "2021-01-01", "2022-01-01"),
        ("2020-01-01", "2022-01-01", "2023-01-01"),
        ("2021-01-01", "2023-01-01", "2024-01-01"),
        ("2022-01-01", "2024-01-01", "2025-01-01"),
        ("2023-01-01", "2025-01-01", "2026-05-08"),
    ]

    model, device = ModelFactory.create_model(
        model_type='transformer',
        ts_feature_dim=22,
        meta_feature_dim=7,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} | 设备: {device}\n")

    results = []

    for idx, (train_start, train_end, val_end) in enumerate(windows):
        print(f"\n{'='*50}")
        print(f">>> 滚动窗口 {idx + 1}/{len(windows)}")
        print(f"    训练: {train_start} ~ {train_end}")
        print(f"    验证: {train_end} ~ {val_end}")
        print(f"{'='*50}")

        train_dataset = OfflineStockDataset(
            data_dir, seq_len=SEQ_LEN, pred_len=PRED_LEN,
            is_train=True, start_date=train_start, end_date=train_end,
        )
        val_dataset = OfflineStockDataset(
            data_dir, seq_len=SEQ_LEN, pred_len=PRED_LEN,
            is_train=False, start_date=train_end, end_date=val_end,
        )
        val_dataset.set_scaler(train_dataset.scaler)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"[跳过] 窗口 {idx + 1} 数据不足")
            continue

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )

        trainer = EchoMarshTrainer(
            model=model,
            device=device,
            checkpoint_dir=checkpoint_dir,
            lr=3e-4 if idx == 0 else 1e-4,
            epochs=50,
            patience=8,
            use_amp=True,
        )

        trainer.fit(train_loader, val_loader)

        best_val = getattr(trainer, 'best_val_loss', float('inf'))
        best_epoch = getattr(trainer, 'best_epoch', 0)
        results.append({
            "Window": f"{train_end} ~ {val_end}",
            "Best_Val_Loss": best_val,
            "Best_Epoch": best_epoch,
        })

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_window_{idx+1}.pt"))

    print("\n" + "=" * 60)
    print("Walk-Forward 完成！结果摘要：")
    for res in results:
        print(f"  {res['Window']:20s} | Loss: {res['Best_Val_Loss']:.4f} (epoch {res['Best_Epoch']})")
    print("=" * 60)


if __name__ == "__main__":
    run_walk_forward_validation()

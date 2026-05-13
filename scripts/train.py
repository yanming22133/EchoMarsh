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

    # 数据路径 — 日线前复权数据
    data_dir = r"f:\EchoMarsh\5.8号\每只股票一个文件\前复权\前复权"
    checkpoint_dir = os.path.join(project_root, "models", "checkpoints")

    # 超参数
    SEQ_LEN = 120          # 日线：120 个交易日 ≈ 6 个月
    PRED_LEN = 5           # 预测未来 5 个交易日收益率
    BATCH_SIZE = 64        # 32→64，每轮步数减半，4070 8G 够用
    NUM_WORKERS = 0        # Windows 8GB 内存必须设为 0，避免 worker 进程复制内存
    EPOCHS = 200
    LR = 1e-4         # 3e-4 → 1e-4 防止梯度爆炸
    PATIENCE = 15

    print(f"[配置] seq_len={SEQ_LEN}, pred_len={PRED_LEN}, "
          f"batch={BATCH_SIZE}, workers={NUM_WORKERS}")

    # 加载数据——严格时间切分，防止未来信息泄漏
    # 训练集: 2010-01 ~ 2024-01  (14年)
    # 验证集: 2024-01 ~ 2026-05  (2年+)
    #
    # 内存约束 (8GB RAM)：全量 5839 只股票样本 ~2000 万 → 远超内存。
    # max_files=200 控制单数据集 ~2.5GB, 两数据集峰值 ~5GB (8GB 内存上限)
    MAX_FILES = 200

    print(f"\n加载训练数据: {data_dir} (max_files={MAX_FILES})")
    train_dataset = OfflineStockDataset(
        data_dir, seq_len=SEQ_LEN, pred_len=PRED_LEN,
        is_train=True, start_date='2010-01-01', end_date='2024-01-01',
        max_files=MAX_FILES,
    )
    train_scaler = train_dataset.scaler  # 保存 scaler 后再释放 dataset
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(NUM_WORKERS > 0),
    )
    del train_dataset

    print(f"\n加载验证数据...")
    val_dataset = OfflineStockDataset(
        data_dir, seq_len=SEQ_LEN, pred_len=PRED_LEN,
        is_train=False, start_date='2024-01-01', end_date='2026-05-08',
        max_files=MAX_FILES,
    )
    val_dataset.set_scaler(train_scaler)

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(NUM_WORKERS > 0),
    )
    del val_dataset

    if len(train_loader) == 0:
        print("[ERROR] 训练集为空，请检查数据路径。")
        return

    # 初始化模型
    model, device = ModelFactory.create_model(
        model_type='transformer',
        ts_feature_dim=22,
        meta_feature_dim=7,
    )

    # 训练器（启用 AMP 混合精度加速）
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

"""
EchoMarsh (鸣泽) 量化系统 — 流式训练 (IterableDataset)
==================================================
不存储全量样本到内存，支持全市场 5839 只股票训练。
"""
import os, sys, pickle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from torch.utils.data import DataLoader
from core.trainer.stream_dataset import StreamStockDataset, create_stream_dataloaders
from models.backbone.model_factory import ModelFactory
from core.trainer.trainer import EchoMarshTrainer


def main():
    data_dir = "/root/autodl-tmp/前复权"
    checkpoint_dir = os.path.join(project_root, "models", "checkpoints")

    SEQ_LEN, PRED_LEN = 120, 5
    BATCH_SIZE = 512          # 4090 24GB，流式可以更大
    NUM_WORKERS = 4           # 多 worker 分片读取文件
    EPOCHS = 200
    LR = 1e-4
    PATIENCE = 20
    INCLUDE_CODES = ('60', '00')   # 主板

    print(f"配置: seq_len={SEQ_LEN}, batch={BATCH_SIZE}, workers={NUM_WORKERS}")

    # 创建流式 DataLoader（自动拟合 scaler）
    train_loader, val_loader = create_stream_dataloaders(
        data_dir=data_dir,
        seq_len=SEQ_LEN, pred_len=PRED_LEN,
        start_date='2010-01-01', end_date='2026-05-08',
        train_end_date='2024-01-01',
        include_codes=INCLUDE_CODES,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        cache_dir="/root/autodl-tmp/cache",
    )

    if train_loader is None:
        print("[ERROR] 训练集为空")
        return

    # 初始化模型
    model, device = ModelFactory.create_model('transformer', ts_feature_dim=36, meta_feature_dim=7)

    # 训练
    trainer = EchoMarshTrainer(model, device, checkpoint_dir, lr=LR, epochs=EPOCHS, patience=PATIENCE, use_amp=True)
    trainer.fit(train_loader, val_loader)

    print("训练完成！最佳模型:", os.path.join(checkpoint_dir, "best_echomarsh_model.pth"))


if __name__ == "__main__":
    main()

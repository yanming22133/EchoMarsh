import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from core.scanner.preprocessor import Preprocessor

class EchoMarshDataset(Dataset):
    def __init__(self, data_dir, sequence_length=60, future_window=15, split='train', train_ratio=0.8):
        """
        :param split: 'train' 或 'val'。按时间顺序切分（严禁随机 shuffle，防止未来数据泄露）
        """
        self.sequence_length = sequence_length
        self.preprocessor = Preprocessor(sequence_length=sequence_length, future_window=future_window)

        self.feature_cols = [f'{c}_norm' for c in self.preprocessor.feature_cols]

        self.samples = []  # (ts_array, meta_array, y)
        self._build_dataset(data_dir, split, train_ratio)

    def _build_dataset(self, data_dir, split, train_ratio):
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        print(f"Found {len(csv_files)} CSV files in {data_dir}.")

        all_samples = []
        for file in csv_files:
            df = self.preprocessor.process_file(file)
            if df is None or len(df) < self.sequence_length:
                continue

            # 确保特征列存在
            missing = [c for c in self.feature_cols if c not in df.columns]
            if missing:
                continue

            features = df[self.feature_cols].values.astype(np.float32)
            labels   = df['target_max_return_15m'].values.astype(np.float32)

            # meta 特征（暂时使用竞价量比等日线级别特征）
            meta_cols = ['auction_ratio']
            meta_vals = []
            for mc in meta_cols:
                if mc in df.columns:
                    v = df[mc].iloc[0]
                    meta_vals.append(float(v) if not np.isnan(v) else 0.0)
                else:
                    meta_vals.append(0.0)
            # 补齐到 7 维
            while len(meta_vals) < 7:
                meta_vals.append(0.0)
            meta_arr = np.array(meta_vals, dtype=np.float32)

            # 按时间顺序生成滑动窗口样本
            for i in range(len(features) - self.sequence_length + 1):
                x  = features[i:i + self.sequence_length]
                y  = labels[i + self.sequence_length - 1]
                if np.isnan(x).any() or np.isnan(y) or np.isinf(x).any():
                    continue
                all_samples.append((x, meta_arr, y))

        # 按时间顺序切分（绝对不能 shuffle！）
        n = len(all_samples)
        split_idx = int(n * train_ratio)
        if split == 'train':
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]

        print(f"[{split}] Dataset built: {len(self.samples)} samples (from {n} total)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, meta, y = self.samples[idx]
        return (
            torch.tensor(x,    dtype=torch.float32),
            torch.tensor(meta, dtype=torch.float32),
            torch.tensor(y,    dtype=torch.float32),
        )


def get_dataloaders(data_dir, batch_size=32, sequence_length=60, train_ratio=0.8):
    train_ds = EchoMarshDataset(data_dir, sequence_length=sequence_length, split='train', train_ratio=train_ratio)
    val_ds   = EchoMarshDataset(data_dir, sequence_length=sequence_length, split='val',   train_ratio=train_ratio)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0) if len(train_ds) else None
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0) if len(val_ds)   else None

    return train_loader, val_loader


class EchoMarshDatasetFromFiles(Dataset):
    """
    支持直接传入文件列表构建数据集。
    用于 Walk-Forward 训练中，按时间窗口切分指定的 CSV 文件子集。
    """
    def __init__(self, file_list, sequence_length=60, future_window=15):
        self.sequence_length = sequence_length
        self.preprocessor = Preprocessor(sequence_length=sequence_length, future_window=future_window)
        self.feature_cols = [f'{c}_norm' for c in self.preprocessor.feature_cols]
        self.samples = []
        self._build(file_list)

    def _build(self, file_list):
        for f in sorted(file_list):
            df = self.preprocessor.process_file(f)
            if df is None or len(df) < self.sequence_length:
                continue
            missing = [c for c in self.feature_cols if c not in df.columns]
            if missing:
                continue
            features = df[self.feature_cols].values.astype(np.float32)
            labels   = df['target_max_return_15m'].values.astype(np.float32)
            meta_arr = np.zeros(7, dtype=np.float32)
            if 'auction_ratio' in df.columns:
                meta_arr[0] = float(df['auction_ratio'].iloc[0])
            for i in range(len(features) - self.sequence_length + 1):
                x = features[i:i + self.sequence_length]
                y = labels[i + self.sequence_length - 1]
                if np.isnan(x).any() or np.isnan(y) or np.isinf(x).any():
                    continue
                self.samples.append((x, meta_arr, y))
        print(f"[FileDataset] Built {len(self.samples)} samples from {len(file_list)} files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, meta, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(meta, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

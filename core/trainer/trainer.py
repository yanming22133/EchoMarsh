import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

class FocalLoss(nn.Module):
    """
    Focal Loss：专门处理极度不平衡的涨停样本（全市场每天约 1% 的票能涨停）
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class EchoMarshTrainer:
    def __init__(self, model, device, checkpoint_dir="models/checkpoints",
                 lr=3e-4, epochs=200, patience=10, use_amp=True):
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.epochs = epochs
        self.patience = patience
        self.use_amp = use_amp and (device.type == 'cuda')

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.reg_loss_fn = nn.HuberLoss(delta=0.03)
        self.cls_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, fused=(device.type == 'cuda'))
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2)

        # AMP（混合精度）梯度缩放器
        self.scaler = torch.amp.GradScaler(device='cuda', enabled=self.use_amp)

    def _compute_loss(self, outputs, targets):
        return_preds   = outputs[:, 0]
        limit_logits   = outputs[:, 1]
        # FIXED: 阈值从 0.095 改为 9.5，Target_Return 单位是百分比(5.0=5%)
        true_limit_up  = (targets > 9.5).float()

        loss_reg = self.reg_loss_fn(return_preds, targets)
        loss_cls = self.cls_loss_fn(limit_logits, true_limit_up)
        return 0.35 * loss_reg + 0.65 * loss_cls, loss_reg.item(), loss_cls.item()

    def _run_epoch(self, dataloader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss = total_reg = total_cls = 0.0
        ctx = torch.enable_grad() if train else torch.no_grad()
        phase = "Train" if train else "Val"
        pbar = tqdm(dataloader, desc=f"  {phase}", leave=False, ncols=100)

        with ctx:
            for ts_batch, meta_batch, y_batch in pbar:
                ts_batch   = ts_batch.to(self.device, non_blocking=True)
                meta_batch = meta_batch.to(self.device, non_blocking=True)
                y_batch    = y_batch.to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad()

                # AMP 混合精度前向
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(ts_batch, meta_batch)
                    loss, l_reg, l_cls = self._compute_loss(outputs, y_batch)

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item()
                total_reg  += l_reg
                total_cls  += l_cls
                pbar.set_postfix(loss=f"{loss.item():.4f}", reg=l_reg, cls=l_cls)

        n = len(dataloader)
        pbar.close()
        return total_loss / n, total_reg / n, total_cls / n

    def fit(self, train_loader, val_loader=None):
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        patience_counter = 0

        print(f"EchoMarsh ConvTransformer Training | Device: {self.device}")
        print(f"Epochs: {self.epochs} | Patience: {self.patience}")
        print("-" * 70)

        for epoch in range(1, self.epochs + 1):
            train_loss, l_reg, l_cls = self._run_epoch(train_loader, train=True)
            self.scheduler.step(epoch)
            lr_now = self.optimizer.param_groups[0]['lr']

            val_str = ""
            monitor_loss = train_loss

            if val_loader:
                val_loss, _, _ = self._run_epoch(val_loader, train=False)
                val_str = f" | Val Loss: {val_loss:.6f}"
                monitor_loss = val_loss

            print(f"Epoch {epoch:03d} | Train: {train_loss:.6f} "
                  f"(Reg={l_reg:.5f} Cls={l_cls:.5f}){val_str} | LR: {lr_now:.6f}")

            if monitor_loss < self.best_val_loss:
                self.best_val_loss = monitor_loss
                self.best_epoch = epoch
                patience_counter = 0
                ckpt_path = os.path.join(self.checkpoint_dir, "best_echomarsh_model.pth")
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  --> Best: {self.best_val_loss:.6f} @ Epoch {self.best_epoch}. Saved.")
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"\nEarly Stop: No improvement for {self.patience} epochs.")
                print(f"Best model: Epoch {self.best_epoch}, Loss={self.best_val_loss:.6f}")
                break

        print("\nTraining complete.")

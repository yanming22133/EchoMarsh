import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

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
                 lr=3e-4, epochs=200, patience=10, use_amp=True, warmup_epochs=5):
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.epochs = epochs
        self.patience = patience
        self.use_amp = use_amp and (device.type == 'cuda')
        self.warmup_epochs = warmup_epochs
        self.base_lr = lr

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.reg_loss_fn = nn.HuberLoss(delta=0.03)
        self.cls_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, fused=(device.type == 'cuda'))
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2)

        # AMP（混合精度）梯度缩放器
        self.scaler = torch.amp.GradScaler(device='cuda', enabled=self.use_amp)

    def _compute_loss(self, outputs, targets):
        """
        outputs: [B, 5] = (1d_ret, 3d_ret, 5d_ret, 5d_max_ret, limit_up_logit)
        targets: [B, 5] = (1d_ret, 3d_ret, 5d_ret, 5d_max_ret, limit_flag)
        """
        # 4 个回归头: 用 HuberLoss 分别计算后平均
        reg_1d = nn.functional.huber_loss(outputs[:, 0], targets[:, 0], delta=0.03)
        reg_3d = nn.functional.huber_loss(outputs[:, 1], targets[:, 1], delta=0.03)
        reg_5d = nn.functional.huber_loss(outputs[:, 2], targets[:, 2], delta=0.03)
        reg_5d_max = nn.functional.huber_loss(outputs[:, 3], targets[:, 3], delta=0.03)
        reg_loss = (reg_1d + reg_3d + reg_5d + reg_5d_max) / 4.0

        # 分类头: FocalLoss on limit-up
        cls_loss = self.cls_loss_fn(outputs[:, 4], targets[:, 4])

        # 加权组合: 回归 0.5 + 分类 0.5
        total = 0.5 * reg_loss + 0.5 * cls_loss
        return total, reg_loss.item(), cls_loss.item()

    def _run_epoch(self, dataloader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss = total_reg = total_cls = 0.0
        n_batches = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        last_report = 0

        with ctx:
            for ts_batch, meta_batch, y_batch in dataloader:
                ts_batch   = ts_batch.to(self.device, non_blocking=True)
                meta_batch = meta_batch.to(self.device, non_blocking=True)
                y_batch    = y_batch.to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(ts_batch, meta_batch)
                    loss, l_reg, l_cls = self._compute_loss(outputs, y_batch)

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item()
                total_reg  += l_reg
                total_cls  += l_cls
                n_batches += 1

                # 每 2000 批打印一次进度
                if n_batches - last_report >= 2000:
                    last_report = n_batches
                    print(f"  {'Train' if train else 'Val'}: {n_batches} batches, "
                          f"loss={total_loss/n_batches:.4f}")

        phase = "Train" if train else "Val"
        avg_loss = total_loss / n_batches if n_batches else 0
        avg_reg = total_reg / n_batches if n_batches else 0
        avg_cls = total_cls / n_batches if n_batches else 0
        print(f"  [{phase}] {n_batches} batches | loss={avg_loss:.6f} reg={avg_reg:.5f} cls={avg_cls:.5f}")
        return avg_loss, avg_reg, avg_cls

    def fit(self, train_loader, val_loader=None):
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        patience_counter = 0

        print(f"EchoMarsh ConvTransformer Training | Device: {self.device}")
        print(f"Epochs: {self.epochs} | Patience: {self.patience}")
        print("-" * 70)

        for epoch in range(1, self.epochs + 1):
            train_loss, l_reg, l_cls = self._run_epoch(train_loader, train=True)

            # LR Warmup: 前 warmup_epochs 轮线性升温到 base_lr
            if epoch <= self.warmup_epochs:
                lr = self.base_lr * epoch / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr
            else:
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

            # 每 5 轮定期保存，崩了也不白练
            if epoch % 5 == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"echomarsh_epoch{epoch}.pth")
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  --> Checkpoint saved: epoch{epoch}")

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

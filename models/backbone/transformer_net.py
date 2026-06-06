import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=240):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ConvBlock(nn.Module):
    """
    1D-CNN 局部特征提取块
    专门捕捉游资在 K 线中留下的"微观博弈指纹"：
    - 连续3根大量K线（主力扫货信号）
    - 快速拉升后的缩量滞涨形态（顶部出货）
    - 分时图上的 V 型反转底部特征
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        # 残差连接 (Residual Connection) 防止深层梯度消失
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # x: [Batch, Channels, Seq_Len]
        return self.conv(x) + self.residual(x)


class EchoMarshConvTransformer(nn.Module):
    def __init__(
        self,
        ts_feature_dim=36,
        meta_feature_dim=7,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    ):
        """
        重型装甲版：CNN-Transformer 混合多模态双塔网络
        对标机构级量化系统，充分榨干 RTX 4070 的算力
        
        :param ts_feature_dim: 时序特征维度 (TA 扩充后 22+)
        :param d_model: 隐层维度 256 (从轻量化的 64 升级)
        :param nhead: 多头注意力头数 8
        :param num_layers: Transformer 深度 4 层 (从 2 层升级)
        :param dim_feedforward: FFN 维度 512
        """
        super().__init__()
        
        # ============================================================
        # 塔 1: CNN-Transformer 高频量价混合塔
        # ============================================================
        
        # Stage A: 1D-CNN 提取局部微观形态指纹
        self.conv_block1 = ConvBlock(ts_feature_dim, d_model // 2, kernel_size=3)
        self.conv_block2 = ConvBlock(d_model // 2, d_model, kernel_size=5)
        
        # Stage B: 位置编码注入
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        
        # Stage C: 深层 Transformer 捕捉全局时序依赖
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',  # GELU 比 ReLU 对 Transformer 更友好
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 注意力池化 (Attention Pooling) 代替简单的平均池化
        self.attn_pool = nn.Linear(d_model, 1)
        
        # ============================================================
        # 塔 2: 属性、情绪与资金背离 MLP 塔
        # ============================================================
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_feature_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )
        
        # ============================================================
        # 融合网络: 双塔特征拼接后压缩到最终预测输出
        # ============================================================
        fusion_dim = d_model + 32
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 5)  # 输出: [收益预测(回归) x4, 涨停概率Logit(分类)]
        )

    def forward(self, ts_tensor, meta_tensor):
        """
        :param ts_tensor: [Batch, Seq_Len, ts_feature_dim]
        :param meta_tensor: [Batch, meta_feature_dim]
        :return: [Batch, 5] -> (1d_ret, 3d_ret, 5d_ret, 5d_max_ret, limit_up_logit)
        """
        # --- 塔 1: CNN 局部特征提取 ---
        # Conv1d 需要 [Batch, Channels, Seq_Len]，先转置
        x = ts_tensor.transpose(1, 2)  # [B, F, S]
        x = self.conv_block1(x)         # [B, d_model/2, S]
        x = self.conv_block2(x)         # [B, d_model, S]
        x = x.transpose(1, 2)           # [B, S, d_model]
        
        # --- 塔 1: 位置编码 + Transformer 全局依赖 ---
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x) # [B, S, d_model]
        
        # --- 注意力加权池化 (Attention Pooling) ---
        attn_weights = torch.softmax(self.attn_pool(x), dim=1)  # [B, S, 1]
        ts_feat = (x * attn_weights).sum(dim=1)                 # [B, d_model]

        # --- 塔 2: Meta 特征压缩 ---
        meta_feat = self.meta_mlp(meta_tensor)  # [B, 32]

        # --- 融合与输出 ---
        fused = torch.cat((ts_feat, meta_feat), dim=1)  # [B, d_model+32]
        output = self.fusion_head(fused)                # [B, 2]
        return output


if __name__ == "__main__":
    model = EchoMarshConvTransformer(ts_feature_dim=32, meta_feature_dim=7)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture: CNN (1D) + Transformer (4L, d=256, h=8)")
    print(f"Total Parameters: {total_params:,}")

    batch_size = 32
    dummy_ts = torch.randn(batch_size, 60, 22)
    dummy_meta = torch.randn(batch_size, 7)

    out = model(dummy_ts, dummy_meta)
    print(f"Input TS:   {dummy_ts.shape}")
    print(f"Input Meta: {dummy_meta.shape}")
    print(f"Output:     {out.shape} (expected [{batch_size}, 5])")

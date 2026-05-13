import torch
import numpy as np
import pandas as pd

class MultimodalPreprocessor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length

    def normalize_zscore(self, val, mean, std):
        if std == 0:
            return 0.0
        return (val - mean) / std

    def stitch_tensors(self, kline_df, meta_dict, sentiment_history):
        """
        核心的张量缝合函数 (Feature Crossing & Stitching)。
        将量价时序数据、财务基本面数据、情绪与资金交叉特征缝合成 PyTorch 可用的多模态张量。
        
        :param kline_df: pandas DataFrame, 包含归一化后的分钟级量价序列 (长度 >= sequence_length)
        :param meta_dict: dict, 包含财务信息(市值, PE)、拉萨席位得分、板块偏离度
        :param sentiment_history: dict, 记录 T 和 T-1 的情绪分及资金流入
        :return: dict 包含 'ts_tensor' (时序张量) 和 'meta_tensor' (属性交叉张量)
        """
        # 1. 构造时序张量 Tensor_TS: [Seq_Len, K-Line_Features]
        feature_cols = [col for col in kline_df.columns if col.endswith('_norm')]
        ts_data = kline_df[feature_cols].values[-self.sequence_length:]
        tensor_ts = torch.tensor(ts_data, dtype=torch.float32)

        # 2. 构造特征交叉 (Cross Features): T-1 情绪 vs T 资金
        sentiment_t_minus_1 = sentiment_history.get('sentiment_T_minus_1', 0.0)
        main_net_inflow_T = sentiment_history.get('main_fund_T_ratio', 0.0)
        
        # 资金与情绪的背离差值 (发酵/收割因子)
        sentiment_fund_divergence = sentiment_t_minus_1 - (main_net_inflow_T / 100.0) # 简易缩放

        # 3. 构造静态/低频属性张量 Tensor_Meta: [Meta_Features]
        # 假设我们已知全市场的均值和标准差用于 Z-Score，这里使用 mock 数值
        market_cap_norm = self.normalize_zscore(meta_dict.get('total_market_cap', 0), mean=100e8, std=50e8)
        pe_norm = self.normalize_zscore(meta_dict.get('pe_ratio', 0), mean=30, std=20)
        lhb_seat_score = meta_dict.get('lhb_seat_score', 0.0)
        sector_divergence = meta_dict.get('sector_divergence', 0.0)
        
        # 缝合所有元特征和交叉特征
        meta_features = [
            market_cap_norm,
            pe_norm,
            lhb_seat_score,
            sector_divergence,
            sentiment_t_minus_1,
            main_net_inflow_T / 100.0,
            sentiment_fund_divergence # 核心杀猪盘/合力指标
        ]
        tensor_meta = torch.tensor(meta_features, dtype=torch.float32)

        return {
            'ts_tensor': tensor_ts,      # shape: [60, num_kline_features]
            'meta_tensor': tensor_meta   # shape: [7]
        }

if __name__ == "__main__":
    # 简单的运行测试
    stitcher = MultimodalPreprocessor()
    
    # Mock K-line data
    mock_kline = pd.DataFrame(np.random.randn(100, 5), columns=['A_norm', 'B_norm', 'C_norm', 'D_norm', 'E_norm'])
    
    # Mock Meta data
    mock_meta = {
        'total_market_cap': 150e8,
        'pe_ratio': 45.0,
        'lhb_seat_score': -0.5, # 拉萨天团减分
        'sector_divergence': 6.5 # 逆势大涨 6.5%
    }
    
    # Mock Sentiment
    mock_sent = {
        'sentiment_T_minus_1': 0.8, # 昨天全网狂吹
        'main_fund_T_ratio': -10.0  # 今天主力净流出 10%
    }
    
    tensors = stitcher.stitch_tensors(mock_kline, mock_meta, mock_sent)
    print(f"TS Tensor Shape: {tensors['ts_tensor'].shape}")
    print(f"Meta Tensor Shape: {tensors['meta_tensor'].shape}")
    print(f"Cross Feature (Sent - Fund): {tensors['meta_tensor'][-1].item():.4f}")

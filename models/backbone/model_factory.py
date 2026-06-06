import torch
import torch.nn as nn
from models.backbone.transformer_net import EchoMarshConvTransformer as EchoMarshTransformer

class ModelFactory:
    @staticmethod
    def create_model(model_type='transformer', **kwargs):
        """
        工厂方法，创建并初始化量化模型，并自动挂载到最佳设备 (CUDA)。
        """
        if model_type == 'transformer':
            # CNN-Transformer 重装架构配置
            model = EchoMarshTransformer(
                ts_feature_dim=kwargs.get('ts_feature_dim', 36),
                meta_feature_dim=kwargs.get('meta_feature_dim', 7),
                d_model=kwargs.get('d_model', 256),
                nhead=kwargs.get('nhead', 8),
                num_layers=kwargs.get('num_layers', 4),
                dim_feedforward=kwargs.get('dim_feedforward', 512),
                dropout=kwargs.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Xavier 初始化，帮助梯度更好地在 Transformer 中流动
        ModelFactory.initialize_weights(model)
        
        # 自动探测并挂载到 RTX 4070 (如果环境有 CUDA)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Model [{model_type}] created and loaded onto {device}.")
        
        return model, device

    @staticmethod
    def initialize_weights(model):
        """
        针对 Transformer 网络的权重初始化最佳实践
        """
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

if __name__ == "__main__":
    # 测试工厂初始化与设备挂载
    print("Testing Model Factory...")
    model, device = ModelFactory.create_model()
    print("Total Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

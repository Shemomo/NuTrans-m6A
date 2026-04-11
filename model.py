import torch
import torch.nn as nn
from transformers import AutoModel

class NuTrans_m6A(nn.Module):
    """
    NuTrans-m6A: A Nucleotide Transformer-based Refinement Model for m6A Prediction.
    
    This model leverages the deep semantic embeddings from the 20th layer of the 
    Nucleotide Transformer and performs further refinement through the final 
    4 encoder layers, followed by a 1D-CNN and a MLP classifier.
    """
    def __init__(self, model_name="InstaDeepAI/nucleotide-transformer-500m-human-ref", dropout_rate=0.3):
        super(NuTrans_m6A, self).__init__()
        
        # 1. 加载预训练模型并提取最后 4 层 Encoder
        # 这种设计允许模型在推理时继续处理特征处理脚本导出的第 20 层隐藏状态
        full_model = AutoModel.from_pretrained(model_name)
        self.last_4_layers = full_model.encoder.layer[-4:] 
        del full_model 
        
        # 2. 卷积特征提取头 (1D-CNN Head)
        # 输入维度 1280 来自 Nucleotide Transformer 的隐藏层维度
        self.conv_head = nn.Sequential(
            nn.Conv1d(in_channels=1280, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) # 全局池化，增强对位点特征的鲁棒性
        )
        
        # 3. 预测器 (Predictor)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1) # 二分类输出 (m6A 位点 vs 非位点)
        )
        
    def forward(self, x):
        """
        Forward pass for NuTrans-m6A.
        Args:
            x (Tensor): Hidden states from the 20th layer of NT, shape (Batch, Seq_len, 1280)
        Returns:
            logits (Tensor): Predicted raw scores, shape (Batch, 1)
        """
        # A. 进一步细化高层语义信息 (Refinement)
        hidden_states = x
        for layer_module in self.last_4_layers:
            hidden_states = layer_module(hidden_states)[0]
            
        # B. 特征降维与空间模式捕捉: [B, L, D] -> [B, D, L]
        feat = hidden_states.transpose(1, 2)
        
        # C. 卷积与全局特征提取
        feat = self.conv_head(feat).squeeze(-1) # 压缩维度至 [Batch, 256]
        
        # D. 分类预测
        logits = self.classifier(feat)
        return logits

# -------------------------------------------------------------------------
# 模型实例化示例
# -------------------------------------------------------------------------
if __name__ == "__main__":
    model = NuTrans_m6A()
    print("NuTrans-m6A 架构初始化成功！")
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f} M")
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # --- [消融] 扩展的 Linear Branch ---
        # 移除了 GRU 分支后，完全依赖线性分支处理所有通道的映射
        self.linear_branch = nn.Linear(self.seq_len, self.pred_len)
        
        # --- [消融] 移除了整个特征变量抗噪通道 (Feature Track) ---
        # 注释掉：self.feature_embedding, self.gru, self.decode_proj
        
        # --- [消融] 移除了融合权重 ---
        # 注释掉：self.combine_weight

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, L, C]
        B, L, C = x_enc.shape
        
        # 1. 归一化 (保留 RevIN，将在后续实验消融)
        means = x_enc.mean(1, keepdim=True)
        x = x_enc - means
        stds = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stds
        
        # --- [消融] 移除了 GRU 分支的 Patching 和 Encoding 过程 ---
        
        # 2. 全通道 Linear 映射
        x = x.permute(0, 2, 1) # 转换维度为 [B, C, L]
        out = self.linear_branch(x) # 线性映射: [B, C, pred_len]
        out = out.permute(0, 2, 1) # 转换回 [B, pred_len, C]
        
        # 3. 反归一化
        out = out * stds + means
        
        return out
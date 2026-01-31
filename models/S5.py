import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        
        # --- Patching 配置 ---
        # 建议 seg_len 取 16 或 24，stride 取 8 或 12 (这就有了 overlap，信息更丰富)
        self.seg_len = getattr(configs, 'seg_len', 16) 
        self.stride = getattr(configs, 'stride', 8)    
        
        # 计算 Patch 的数量
        self.patch_num = (self.seq_len - self.seg_len) // self.stride + 1
        
        # --- Track A: Anchor Track (Trend) ---
        # 极简 Linear，负责“稳”
        self.target_linear = nn.Linear(self.seq_len, self.pred_len)
        
        # --- Track B: Feature Track (Dynamics) ---
        # Patch-GRU，负责“准”
        self.feature_embedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 双向 GRU 捕捉 Patch 间的前后文关系
        self.gru = nn.GRU(input_size=self.d_model, 
                          hidden_size=self.d_model, 
                          num_layers=1, 
                          batch_first=True, 
                          bidirectional=True)
        
        # Flatten Head: 将所有 Patch 的 GRU 输出展平，通过 Linear 映射到预测
        # 这种设计比只取最后一个 hidden state 信息量更大
        self.head = nn.Flatten()
        self.feature_proj = nn.Linear(self.patch_num * self.d_model * 2, self.pred_len)
        
        # --- Fusion ---
        # 这种初始化保证开始时各占 50%，让梯度自己去学偏好
        self.combine_weight = nn.Parameter(torch.ones(2))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 输入形状: [Batch, Seq_Len, Channels]
        B, L, C = x_enc.shape
        
        # =============================================================
        # 1. Channel Independence & RevIN (核心修改)
        # =============================================================
        # 变换为 [Batch * Channels, Seq_Len, 1]，把多变量拆成多个单变量处理
        x = x_enc.permute(0, 2, 1).contiguous().reshape(B * C, L)
        
        # 归一化 (Instance Normalization)
        means = x.mean(1, keepdim=True)
        x = x - means
        stds = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stds
        
        # =============================================================
        # 2. Track A: Anchor Track (Global Trend)
        # =============================================================
        # 输入: [B*C, L] -> 输出: [B*C, Pred_Len]
        trend_pred = self.target_linear(x)
        
        # =============================================================
        # 3. Track B: Feature Track (Local Dynamics)
        # =============================================================
        # Patching: 使用 unfold 进行切片
        # 输出形状: [B*C, patch_num, seg_len]
        patches = x.unfold(dimension=1, size=self.seg_len, step=self.stride)
        
        # Embedding: [B*C, patch_num, d_model]
        enc_out = self.feature_embedding(patches)
        
        # GRU Modeling: [B*C, patch_num, d_model * 2]
        gru_out, _ = self.gru(enc_out)
        
        # Projection: Flatten -> Linear
        feat_out = self.head(gru_out) # [B*C, patch_num * d_model * 2]
        dynamic_pred = self.feature_proj(feat_out) # [B*C, Pred_Len]
        
        # =============================================================
        # 4. Gated Fusion & Output
        # =============================================================
        w = torch.softmax(self.combine_weight, dim=0)
        
        # 融合
        final_pred = w[0] * trend_pred + w[1] * dynamic_pred
        
        # 反归一化 (Denormalization)
        final_pred = final_pred * stds + means
        
        # 重塑回多变量形状: [Batch, Pred_Len, Channels]
        final_pred = final_pred.reshape(B, C, self.pred_len).permute(0, 2, 1)
        
        return final_pred
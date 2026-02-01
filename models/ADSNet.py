import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.seg_len = getattr(configs, 'seg_len', 24)
        
        # 1. 目标变量专用通道 (Anchor Track) - 学习自相关性
        self.target_linear = nn.Linear(self.seq_len, self.pred_len)
        
        # 2. 特征变量抗噪通道 (Feature Track)
        # 先用一个线性层过滤噪声：(seg_len) -> (d_model)
        self.feature_embedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )
        
        # 使用双向 GRU 增强对 720 步全局信息的捕获
        self.gru = nn.GRU(input_size=self.d_model, 
                          hidden_size=self.d_model, 
                          num_layers=1, 
                          batch_first=True, 
                          bidirectional=True)
        
        # 3. 解码映射：将 GRU 状态直接映射到 pred_len，避开递归
        self.decode_proj = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.pred_len)
        )
        
        # 4. 融合权重（可学习）
        self.combine_weight = nn.Parameter(torch.ones(2))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, L, C]
        B, L, C = x_enc.shape
        
        # --- 分支一：纯净目标变量轨迹 ---
        # 假设最后一列是目标变量 (常见设置)
        target = x_enc[:, :, -1] # [B, L]
        target_pred = self.target_linear(target).unsqueeze(-1) # [B, pred_len, 1]
        
        # --- 分支二：加噪特征抗噪轨迹 ---
        # 1. 归一化（防止噪声幅值过大）
        means = x_enc.mean(1, keepdim=True)
        x = x_enc - means
        stds = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stds
        
        # 2. Patching & Embedding
        # 将 [B, L, C] 转为 CI 模式处理 [B*C, L, 1]
        x = x.permute(0, 2, 1).reshape(B * C, L, 1)
        if L % self.seg_len != 0:
            pad_len = self.seg_len - (L % self.seg_len)
            x = torch.cat([x, x[:, -pad_len:, :]], dim=1)
        
        x = x.reshape(B * C, -1, self.seg_len) # [B*C, Seg_Num, Seg_Len]
        x = self.feature_embedding(x) # [B*C, Seg_Num, d_model]
        
        # 3. GRU Encoding
        _, h_n = self.gru(x) # h_n: [2, B*C, d_model]
        h_n = h_n.permute(1, 0, 2).reshape(B * C, -1) # [B*C, 2*d_model]
        
        # 4. Direct Projection (解决 720 步失效的关键：不递归，直接出结果)
        feat_pred = self.decode_proj(h_n) # [B*C, pred_len]
        feat_pred = feat_pred.reshape(B, C, self.pred_len).permute(0, 2, 1) # [B, pred_len, C]
        
        # --- 最终融合 ---
        # 反归一化
        feat_pred = feat_pred * stds + means
        
        # 融合目标变量分支和特征分支
        # 目标变量使用权重融合，其他特征列保留特征分支结果
        w = torch.softmax(self.combine_weight, dim=0)
        final_target = feat_pred[:, :, -1:]
        
        out = torch.cat([feat_pred[:, :, :-1], final_target], dim=-1)
        
        return out
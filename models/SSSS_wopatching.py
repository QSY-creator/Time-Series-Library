import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        
        # --- [消融] 移除了 patching 相关的 seg_len 属性 ---
        # self.seg_len = getattr(configs, 'seg_len', 24)
        
        # 1. 目标变量专用通道 (Anchor Track)
        self.target_linear = nn.Linear(self.seq_len, self.pred_len)
        
        # 2. 特征变量抗噪通道 (Feature Track)
        # --- [消融] 移除了 Patching，直接对单步（维度为1）进行 Embedding ---
        self.feature_embedding = nn.Sequential(
            nn.Linear(1, self.d_model), # 输入维度从 seg_len 变为单步长 1
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )
        
        # 使用双向 GRU (由于没有 patching，这里处理的序列长度将从 seg_num 变为 L)
        self.gru = nn.GRU(input_size=self.d_model, 
                          hidden_size=self.d_model, 
                          num_layers=1, 
                          batch_first=True, 
                          bidirectional=True)
        
        # 3. 解码映射
        self.decode_proj = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.pred_len)
        )
        
        # 4. 融合权重
        self.combine_weight = nn.Parameter(torch.ones(2))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, L, C]
        B, L, C = x_enc.shape
        
        # --- 分支一：纯净目标变量轨迹 ---
        target = x_enc[:, :, -1] # [B, L]
        target_pred = self.target_linear(target).unsqueeze(-1) # [B, pred_len, 1]
        
        # --- 分支二：加噪特征抗噪轨迹 ---
        # 1. 归一化 (RevIN) 恢复使用
        means = x_enc.mean(1, keepdim=True)
        x = x_enc - means
        stds = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stds
        
        # 2. Embedding 
        # 依然保留通道独立性 (Channel Independence)
        x = x.permute(0, 2, 1).reshape(B * C, L, 1) # 形状: [B*C, L, 1]
        
        # --- [消融] 移除了 Padding 和 Reshape (Patching) 过程 ---
        # 直接使用全序列长 L 进行映射
        x = self.feature_embedding(x) # 形状: [B*C, L, d_model]
        
        # 3. GRU Encoding
        _, h_n = self.gru(x) # h_n: [2, B*C, d_model]
        h_n = h_n.permute(1, 0, 2).reshape(B * C, -1) # [B*C, 2*d_model]
        
        # 4. Direct Projection 
        feat_pred = self.decode_proj(h_n) # [B*C, pred_len]
        feat_pred = feat_pred.reshape(B, C, self.pred_len).permute(0, 2, 1) # [B, pred_len, C]
        
        # --- 最终融合 ---
        # 反归一化
        feat_pred = feat_pred * stds + means
        
        # 融合
        w = torch.softmax(self.combine_weight, dim=0)
        final_target = w[0] * target_pred + w[1] * feat_pred[:, :, -1:]
        
        out = torch.cat([feat_pred[:, :, :-1], final_target], dim=-1)
        
        return out
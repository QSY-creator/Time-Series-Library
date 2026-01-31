import torch
import torch.nn as nn

class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)
        # Padding on both ends to keep size consistent
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.seg_len = getattr(configs, 'seg_len', 24)
        
        # 1. Decomposition
        kernel_size = 25
        self.decompsition = SeriesDecomp(kernel_size)

        # 2. Trend Branch (Simple Linear)
        # 处理长期趋势，不需要复杂的非线性
        self.trend_linear = nn.Linear(self.seq_len, self.pred_len)
        self.trend_linear.weight = nn.Parameter(
            (1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))

        # 3. Seasonal Branch (Your SegRNN/GRU logic)
        # 处理去除趋势后的高频/周期性部分
        self.feature_embedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )
        
        self.gru = nn.GRU(input_size=self.d_model, 
                          hidden_size=self.d_model, 
                          num_layers=1, 
                          batch_first=True, 
                          bidirectional=True)
        
        # Projection for GRU output
        self.decode_proj = nn.Linear(self.d_model * 2, self.pred_len) # Bi-GRU * 2
        
        # 4. Gated Fusion (改进点：动态门控)
        # 即使是简单的融合，加入一个基于 Input 的 Gate 也能提升性能
        # 我们根据 Seasonal 部分的输入特征来决定多大程度上信任 GRU
        self.gate_layer = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len // 4),
            nn.ReLU(),
            nn.Linear(self.seq_len // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, L, C]
        B, L, C = x_enc.shape

        # Step 1: Decomposition
        seasonal_init, trend_init = self.decompsition(x_enc)
        
        # Step 2: Trend Forecasting
        # Permute for Linear layer: [B, L, C] -> [B, C, L] -> Linear -> [B, C, P] -> [B, P, C]
        trend_output = self.trend_linear(trend_init.permute(0, 2, 1)).permute(0, 2, 1)

        # Step 3: Seasonal Forecasting (GRU Path)
        # Instance Norm (RevIN 思想) 仅应用于 Seasonal 部分，保持零均值假设
        means = seasonal_init.mean(1, keepdim=True)
        x = seasonal_init - means
        stds = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stds

        # Segmentation Logic
        # Reshape to [B*C, Num_Segs, Seg_Len]
        # (This implements Channel Independence implicitly by merging B and C)
        x = x.permute(0, 2, 1).reshape(B * C, L, 1)
        if L % self.seg_len != 0:
            pad_len = self.seg_len - (L % self.seg_len)
            x = torch.cat([x, x[:, -pad_len:, :]], dim=1)
        
        num_segs = x.shape[1] // self.seg_len
        x = x.reshape(B * C, num_segs, self.seg_len)
        
        # Embedding & GRU
        x = self.feature_embedding(x) # [B*C, N_segs, d_model]
        _, h_n = self.gru(x)          # h_n: [2, B*C, d_model] (Bidirectional)
        
        # Flatten bidirectional hidden states
        h_n = h_n.permute(1, 0, 2).reshape(B * C, -1) # [B*C, 2*d_model]
        
        # Decode
        seasonal_output = self.decode_proj(h_n) # [B*C, pred_len]
        seasonal_output = seasonal_output.reshape(B, C, self.pred_len).permute(0, 2, 1) # [B, P, C]
        
        # De-Normalization
        seasonal_output = seasonal_output * stds + means

        # Step 4: Gated Fusion
        # 计算 Gate 权重 (基于原始 Seasonal 输入)
        # [B, L, C] -> [B, C, L] -> Gate -> [B, C, 1]
        gate = self.gate_layer(seasonal_init.permute(0, 2, 1)).permute(0, 2, 1) 
        
        # 最终融合：Gate * Seasonal + (1-Gate) * Trend
        # 如果 Gate 接近 1，说明模型认为局部波动更重要；接近 0 说明主要是趋势
        final_output = gate * seasonal_output + (1 - gate) * trend_output
        
        return final_output # [B, P, C]
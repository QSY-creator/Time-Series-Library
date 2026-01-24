import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 复用 RevIN (来自第一个模型，处理分布漂移) ---
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

# --- 2. 复用 MedianDecomp (来自第一个模型，抗噪能力强) ---
class MedianDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(MedianDecomp, self).__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        x = x.permute(0, 2, 1) # [B, C, L]
        x_pad = F.pad(x, (self.padding, self.padding), mode='replicate')
        x_unfolded = x_pad.unfold(dimension=2, size=self.kernel_size, step=1)
        trend = x_unfolded.median(dim=-1)[0] # 鲁棒的趋势提取
        trend = trend.permute(0, 2, 1) # [B, L, C]
        seasonal = x.permute(0, 2, 1) - trend
        return seasonal, trend

# --- 3. 新设计的模型核心：MSDL ---
class Model(nn.Module):
    """
    Multi-Scale Decomposed Linear (MSDL)
    结合了 DLinear 的稳健 Trend 处理和 TimeMixer 的多尺度 Seasonal 处理。
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # 1. 归一化模块 (Model 1 优势)
        self.revin = RevIN(self.enc_in)
        
        # 2. 分解模块 (Model 1 优势 - 抗 Dropout)
        self.decomposition = MedianDecomp(configs.moving_avg)
        
        # 3. Trend 分支 (Model 1 策略 - 保持简单稳定)
        # 趋势项通常包含低频信息，用单个线性层捕捉长距离依赖最稳健，不易过拟合
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))

        # 4. Seasonal 分支 (Model 2 策略 - 多尺度感知)
        # 季节项/高频项包含细节，使用多尺度处理来提升短期预测精度
        self.down_sampling_layers = torch.nn.ModuleList()
        self.seasonal_layers = torch.nn.ModuleList()
        
        # 定义尺度：[1, 2, 4] 代表 原始分辨率, 1/2分辨率, 1/4分辨率
        # 这里的 scales 可以根据需求调整，例如 [1, 2] 或 [1, 2, 4]
        self.scales = [1, 2] 
        
        for scale in self.scales:
            # 定义下采样层 (Scale=1 时不做操作)
            if scale == 1:
                self.down_sampling_layers.append(nn.Identity())
                input_len = self.seq_len
            else:
                # 使用 AvgPool 进行下采样，类似于 Model 2 的处理
                self.down_sampling_layers.append(nn.AvgPool1d(kernel_size=scale, stride=scale))
                input_len = self.seq_len // scale
            
            # 为每个尺度定义一个 Linear 层
            # 注意：无论输入尺度如何，输出都必须映射回 pred_len
            self.seasonal_layers.append(nn.Linear(input_len, self.pred_len))
            
        # 5. 可选：Seasonal 分支的 Dropout (来自 Model 1 的优化)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Input_Length, Channel]
        
        # Step 1: RevIN Normalization
        x = self.revin(x_enc, 'norm')

        # Step 2: Decomposition
        # seasonal: 高频/细节, trend: 低频/趋势
        seasonal_init, trend_init = self.decomposition(x)
        
        # Step 3: Trend Prediction (单尺度，强调长期稳定性)
        # [B, L, C] -> [B, C, L] for Linear Layer
        trend_init = trend_init.permute(0, 2, 1)
        trend_output = self.linear_trend(trend_init) # -> [B, C, Pred_Len]
        trend_output = trend_output.permute(0, 2, 1) # -> [B, Pred_Len, C]

        # Step 4: Seasonal Prediction (多尺度，强调短期细节)
        seasonal_init = seasonal_init.permute(0, 2, 1) # [B, C, L]
        seasonal_output_sum = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len], 
                                          device=seasonal_init.device)
        
        for i, scale in enumerate(self.scales):
            # A. 下采样
            # [B, C, L] -> [B, C, L // scale]
            s_input = self.down_sampling_layers[i](seasonal_init)
            
            # B. 线性映射
            # Linear 作用在最后一维: [B, C, L_down] -> [B, C, Pred_Len]
            s_out = self.seasonal_layers[i](s_input)
            
            # C. 累加不同尺度的预测结果
            seasonal_output_sum += s_out

        # 对多尺度结果取平均或直接使用 (这里选择直接累加，让模型自己学权重)
        seasonal_output = self.dropout(seasonal_output_sum)
        seasonal_output = seasonal_output.permute(0, 2, 1) # -> [B, Pred_Len, C]

        # Step 5: Final Summation
        x = trend_output + seasonal_output

        # Step 6: RevIN Denormalization
        x = self.revin(x, 'denorm')

        return x
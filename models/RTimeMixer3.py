import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# --- 1. RevIN: 处理 Distribution Drift (不可或缺) ---
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

# --- 2. MedianDecomp: 鲁棒的趋势分离 ---
class MedianDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(MedianDecomp, self).__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        x = x.permute(0, 2, 1) # [B, C, L]
        x_pad = F.pad(x, (self.padding, self.padding), mode='replicate')
        x_unfolded = x_pad.unfold(dimension=2, size=self.kernel_size, step=1)
        # Median 对异常值不敏感，能提取出不受噪声尖峰影响的 Trend
        trend = x_unfolded.median(dim=-1)[0]
        trend = trend.permute(0, 2, 1) # [B, L, C]
        seasonal = x.permute(0, 2, 1) - trend
        return seasonal, trend

# --- 3. (新) FourierFilter: 频域滤波模块 ---
class FourierFilter(nn.Module):
    """
    借鉴 FITS 和 FreTS 的思想：
    在频域进行滤波比在时域更有效，特别是针对高斯噪声和随机 Dropout。
    高斯噪声在频域表现为均匀分布的底噪，而信号集中在特定频率。
    通过学习复数权重，模型可以自动抑制噪声频率。
    """
    def __init__(self, input_len, pred_len, enc_in, dropout=0.1):
        super(FourierFilter, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        
        # FFT 后的长度 (RFFT 仅保留一半频率，加直流分量)
        self.freq_len = input_len // 2 + 1
        
        # 频域滤波器：使用复数权重
        # 这里的 Scale 因子是为了保持梯度稳定
        self.scale = 0.02
        self.complex_weight = nn.Parameter(
            self.scale * torch.randn(enc_in, self.freq_len, 2, dtype=torch.float32)
        )
        
        # 最后的时域映射层
        self.linear = nn.Linear(input_len, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        x = x.permute(0, 2, 1) # [B, C, L]

        # 1. 转到频域 (RFFT)
        # x_fft: [B, C, Freq_Len] (Complex64)
        x_fft = torch.fft.rfft(x, dim=-1)
        
        # 2. 频域加权 (Filtering)
        # 将复数权重转为 complex tensor
        weight = torch.view_as_complex(self.complex_weight) # [C, Freq_Len]
        
        # 元素级乘法：相当于在频域应用一个自适应滤波器
        # 噪声通常是高频且杂乱的，模型会学到把对应的频率权重降低
        x_fft_filtered = x_fft * weight
        
        # 3. 转回时域 (IRFFT)
        x_filtered = torch.fft.irfft(x_fft_filtered, n=self.input_len, dim=-1)
        
        # 4. 预测 (Projection)
        # 此时的 x_filtered 是经过“净化”的时序数据
        x_out = self.dropout(self.linear(x_filtered)) # [B, C, Pred_Len]
        
        return x_out.permute(0, 2, 1) # [B, Pred_Len, C]

# --- 4. 主模型 ---
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # 1. RevIN: 解决 Drift
        self.revin = RevIN(self.enc_in)
        
        # 2. MedianDecomp: 解决 Trend 受到噪声干扰的问题
        # 确保 kernel_size 为奇数
        kernel_size = configs.moving_avg if configs.moving_avg % 2 == 1 else configs.moving_avg + 1
        self.decomposition = MedianDecomp(kernel_size)
        
        # 3. Trend Branch: 简单的 Linear
        # 趋势项不需要复杂的滤波，直接映射最稳健
        self.trend_linear = nn.Linear(self.seq_len, self.pred_len)
        self.trend_linear.weight = nn.Parameter(
            (1/self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # 4. Seasonal Branch: 升级为 FourierFilter
        # 专门处理含噪的 Seasonal 数据
        self.seasonal_filter = FourierFilter(self.seq_len, self.pred_len, self.enc_in, configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Input_Len, Channel]
        
        # Step 1: RevIN
        x = self.revin(x_enc, 'norm')

        # Step 2: Decomposition
        # seasonal 包含噪声和周期性，trend 包含趋势
        seasonal_init, trend_init = self.decomposition(x)
        
        # Step 3: Trend Prediction
        trend_init = trend_init.permute(0, 2, 1)
        trend_output = self.trend_linear(trend_init)
        trend_output = trend_output.permute(0, 2, 1)

        # Step 4: Seasonal Prediction (Frequency Domain Denoising)
        # 这里发生了魔法：在频域去除高斯噪声和 Dropout 带来的高频伪影
        seasonal_output = self.seasonal_filter(seasonal_init)

        # Step 5: Combine
        x = trend_output + seasonal_output
        x = self.revin(x, 'denorm')

        return x
import torch
import torch.nn as nn
import torch.fft

class MovingAverage(nn.Module):
    """
    多尺度移动平均模块，用于提取趋势项。
    """
    def __init__(self, kernel_size, stride):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    """
    序列分解模块
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class FrequencyEnhancedBlock(nn.Module):
    """
    SuperLinear 核心思想：频域增强模块
    """
    def __init__(self, seq_len, pred_len, enc_in):
        super(FrequencyEnhancedBlock, self).__init__()
        self.pred_len = pred_len
        # rfft 输出长度为 seq_len//2 + 1
        self.freq_len = seq_len // 2 + 1
        
        # 频域权重矩阵 (模拟复数操作)
        self.frequency_weight = nn.Parameter(torch.randn(self.freq_len, enc_in, 2) * 0.02)
        
        # 时域映射层
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        B, L, C = x.shape
        
        # 1. 变换到频域
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho') # [B, Freq_Len, C]
        
        # 2. 频域滤波
        weight = torch.view_as_complex(self.frequency_weight) # [Freq_Len, C]
        x_fft_enhanced = x_fft * weight.unsqueeze(0)
        
        # 3. 变换回时域
        x_enhanced = torch.fft.irfft(x_fft_enhanced, n=L, dim=1, norm='ortho')
        
        # 4. 线性映射
        x_out = self.linear(x_enhanced.permute(0, 2, 1)).permute(0, 2, 1)
        
        return x_out

class Model(nn.Module):
    """
    Super_DP: Frequency-Enhanced Decomposed-Pad Model
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # Decomposition Kernel Size
        kernel_size = 25
        self.decompsition = SeriesDecomp(kernel_size)
        
        # Seasonal Branch: Super (Frequency) + Linear (Time)
        self.seasonal_freq_block = FrequencyEnhancedBlock(self.seq_len, self.pred_len, self.enc_in)
        self.seasonal_linear = nn.Linear(self.seq_len, self.pred_len)

        # Trend Branch
        self.trend_linear = nn.Linear(self.seq_len, self.pred_len)
        
        # RevIN
        self.revin = True
        if self.revin:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))

    def _get_statistics(self, x):
        dim2reduce = [1]
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + 1e-5).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.revin:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.revin:
            x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
        x = x * self.stdev
        x = x + self.mean
        return x

    # 修改重点：接收标准参数 x_enc, x_mark_enc, x_dec, x_mark_dec
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        
        # 这里的 x_enc 就是输入序列
        x = x_enc 

        # 1. RevIN
        self._get_statistics(x)
        x = self._normalize(x)

        # 2. Decomposition
        seasonal_init, trend_init = self.decompsition(x)
        
        # 3. Trend Prediction
        trend_output = self.trend_linear(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 4. Seasonal Prediction (Fusion)
        seasonal_freq = self.seasonal_freq_block(seasonal_init)
        seasonal_time = self.seasonal_linear(seasonal_init.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal_output = seasonal_freq + seasonal_time

        # 5. Synthesis
        x_out = seasonal_output + trend_output

        # 6. RevIN Denormalize
        x_out = self._denormalize(x_out)

        return x_out[:, -self.pred_len:, :]
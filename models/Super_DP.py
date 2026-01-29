import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class MovingAverage(nn.Module):
    """
    多尺度移动平均模块，用于提取趋势项。
    相比单一kernel，这里支持加权。
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
    序列分解模块：将输入分解为 Seasonal 和 Trend
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
    SuperLinear 核心改进：频域增强模块
    在频域中应用可学习的滤波器，捕捉全局周期性依赖。
    """
    def __init__(self, seq_len, pred_len, enc_in, decomposition=False):
        super(FrequencyEnhancedBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomposition = decomposition
        
        # 频域线性层：处理 RFFT 后的复数特征 (Real and Imag parts)
        # rfft 输出长度为 seq_len//2 + 1
        self.freq_len = seq_len // 2 + 1
        
        # 这是一个针对频域的权重矩阵，实现“Super”级别的全局交互
        # 使用 Complex Linear 的思想，但为了兼容性拆分为实部虚部计算
        self.frequency_weight = nn.Parameter(torch.randn(self.freq_len, enc_in, 2))
        self.scale = 0.02
        
        # 时域映射层
        self.linear = nn.Linear(seq_len, pred_len)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        B, L, C = x.shape
        
        # 1. 变换到频域
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho') # [B, Freq_Len, C]
        
        # 2. 频域滤波/加权 (Element-wise multiplication with learnable weights)
        # 将复数视为 (Real, Imag)
        weight = torch.view_as_complex(self.frequency_weight) # [Freq_Len, C]
        
        # 广播乘法：让每个频率分量都有独立的权重调整
        # 这里模拟了 SuperLinear 对频率特征的精细化处理
        x_fft_enhanced = x_fft * weight.unsqueeze(0)
        
        # 3. 变换回时域
        x_enhanced = torch.fft.irfft(x_fft_enhanced, n=L, dim=1, norm='ortho')
        
        # 4. 残差连接 + 线性映射到预测长度
        x_out = self.linear(x_enhanced.permute(0, 2, 1)).permute(0, 2, 1)
        
        return x_out

class Model(nn.Module):
    """
    Super-DP (Super Decomposed-Pad) Model
    结合了 DLinear 的分解稳定性与 SuperLinear 的频域建模能力。
    并引入 RevIN 解决分布偏移。
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # 1. Decomposition Kernel Size
        # 动态选择 kernel size，通常取序列长度的 1/4 或 25 左右
        kernel_size = 25
        self.decompsition = SeriesDecomp(kernel_size)
        
        # 2. Frequency Enhanced Seasonal Branch (Super part)
        # 相比简单的 Linear，增加了频域处理，捕捉周期性
        self.seasonal_freq_block = FrequencyEnhancedBlock(self.seq_len, self.pred_len, self.enc_in)
        
        # 保留一个纯线性的 Seasonal 分支作为补充 (DLinear part)
        self.seasonal_linear = nn.Linear(self.seq_len, self.pred_len)

        # 3. Trend Branch (通常 Trend 比较平滑，简单的 Linear 效果最好)
        self.trend_linear = nn.Linear(self.seq_len, self.pred_len)
        
        # 4. RevIN (Reversible Instance Normalization)
        # 这种实现方式不需要外部依赖，内置于模型中
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

    def forward(self, x):
        # x: [Batch, Input_Length, Channel]
        
        # 1. RevIN Normalization
        self._get_statistics(x)
        x = self._normalize(x)

        # 2. Decomposition
        seasonal_init, trend_init = self.decompsition(x)
        
        # 3. Trend Prediction (Direct Linear Mapping)
        # Permute needed for Linear layer: [B, C, L] -> [B, C, Pred]
        trend_output = self.trend_linear(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 4. Seasonal Prediction (Super-Linear Fusion)
        # 分支 A: 频域增强 (捕捉全局周期)
        seasonal_freq = self.seasonal_freq_block(seasonal_init)
        
        # 分支 B: 时域线性 (捕捉局部波动)
        seasonal_time = self.seasonal_linear(seasonal_init.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 融合机制：简单的加和通常最鲁棒
        seasonal_output = seasonal_freq + seasonal_time

        # 5. Final Synthesis
        x_out = seasonal_output + trend_output

        # 6. RevIN Denormalization
        x_out = self._denormalize(x_out)

        return x_out[:, -self.pred_len:, :] # Ensure exact output length
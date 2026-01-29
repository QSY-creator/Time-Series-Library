import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN)
    用于解决时序预测中的分布偏移问题 (Distribution Shift)
    """
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

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
        x = x * self.stdev
        x = x + self.mean
        return x

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x


class SeriesDecomp(nn.Module):
    """
    序列分解模块：将序列分解为 Trend（趋势）和 Seasonal（季节/残差）
    基于移动平均 (Moving Average)
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.padding, 1)
        end = x[:, -1:, :].repeat(1, self.padding, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        
        x_trend = self.moving_avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend
        return x_seasonal, x_trend


class SpectralLinear(nn.Module):
    """
    SuperLinear 核心改进：频域线性层
    在频域中进行线性映射，能更有效地捕捉全局周期性依赖
    """
    def __init__(self, seq_len, pred_len, enc_in):
        super(SpectralLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 只取一半频率 (RFFT性质)
        self.freq_len = seq_len // 2 + 1
        self.pred_freq_len = pred_len // 2 + 1
        
        # 频域权重：实部和虚部
        # 将输入频率映射到输出频率
        self.weight_real = nn.Parameter(torch.randn(self.freq_len, self.pred_freq_len, enc_in))
        self.weight_imag = nn.Parameter(torch.randn(self.freq_len, self.pred_freq_len, enc_in))

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        
        # 1. 转换到频域 (RFFT)
        # x_f: [Batch, Freq_Len, Channel] (Complex64)
        x_f = torch.fft.rfft(x, dim=1)
        
        # 2. 频域线性映射 (Super-Resolution in Frequency)
        # 使用爱因斯坦求和约定进行高效计算
        # 注意：这里我们简化为 Frequency Global Mixing
        # 如果输入输出长度不一致，我们需要在频域做重采样或映射
        
        # 简单策略：在频域对每个Channel独立进行全连接映射
        # Real part calculation
        o_real = torch.einsum('bfc,foc->boc', x_f.real, self.weight_real) - \
                 torch.einsum('bfc,foc->boc', x_f.imag, self.weight_imag)
        
        # Imag part calculation
        o_imag = torch.einsum('bfc,foc->boc', x_f.imag, self.weight_real) + \
                 torch.einsum('bfc,foc->boc', x_f.real, self.weight_imag)
                 
        x_f_out = torch.complex(o_real, o_imag)
        
        # 3. 转换回时域 (IRFFT)
        # 指定输出长度为 pred_len
        x_out = torch.fft.irfft(x_f_out, n=self.pred_len, dim=1)
        
        return x_out


class Model(nn.Module):
    """
    DP_RTM (Dual Path Robust Time Model) - SuperLinear Enhanced Version
    
    结构特点：
    1. RevIN 归一化，抵消分布偏移。
    2. Series Decomposition 将序列分为 Trend 和 Seasonal。
    3. Dual Path 处理：
       - Trend Path: 简单的线性映射 (Linear)，拟合低频趋势。
       - Seasonal Path: 谱增强线性层 (SpectralLinear)，在频域拟合高频周期。
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # 1. Reversible Instance Normalization
        self.revin = RevIN(self.enc_in)
        
        # 2. Series Decomposition
        # kernel_size 通常取 25 或根据数据周期设定，这里取中庸值
        self.decomposition = SeriesDecomp(25)
        
        # 3. Path 1: Seasonal (High Frequency) - 使用 SuperLinear 频域增强
        # 如果为了极致速度，可以使用普通 Linear；为了性能，使用 SpectralLinear
        self.seasonal_projector = SpectralLinear(self.seq_len, self.pred_len, self.enc_in)
        
        # 备选：如果显存受限，可以用 Time-Domain Linear 替代上面的 SpectralLinear
        # self.seasonal_projector = nn.Linear(self.seq_len, self.pred_len)
        # self.seasonal_projector.weight.data.normal_(0, 1/self.seq_len)

        # 4. Path 2: Trend (Low Frequency) - 使用普通 Linear
        self.trend_projector = nn.Linear(self.seq_len, self.pred_len)
        
        # 初始化 Trend 层权重，使其偏向于更平滑的延续
        self.trend_projector.weight.data.normal_(0, 1/self.seq_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Input_Len, Channel]
        
        # 1. RevIN Normalization
        x_enc = self.revin(x_enc, 'norm')

        # 2. Decomposition
        seasonal_init, trend_init = self.decomposition(x_enc)
        
        # 3. Dual Path Processing
        
        # Path 1: Seasonal (Frequency Domain Processing)
        # SpectralLinear 处理通道独立性，同时捕捉全局频率特征
        seasonal_output = self.seasonal_projector(seasonal_init)
        
        # Path 2: Trend (Time Domain Processing)
        # 对 Trend 做简单的线性映射 (Channel Independent 方式)
        # permute 为 [Batch, Channel, Seq_Len] 以通过 Linear 层
        trend_output = self.trend_projector(trend_init.permute(0, 2, 1)).permute(0, 2, 1)

        # 4. Summation
        x_out = seasonal_output + trend_output

        # 5. RevIN Denormalization
        x_out = self.revin(x_out, 'denorm')

        return x_out[:, -self.pred_len:, :] # 确保输出长度对齐
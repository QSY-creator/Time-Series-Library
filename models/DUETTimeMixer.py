import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# --- 1. RevIN (保持不变，抗 Drift 的基石) ---
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

# --- 2. MedianDecomp (保持不变，抗离群点分解) ---
class MedianDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(MedianDecomp, self).__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        # x: [B, L, C] -> [B, C, L] for unfolding
        x = x.permute(0, 2, 1) 
        x_pad = F.pad(x, (self.padding, self.padding), mode='replicate')
        x_unfolded = x_pad.unfold(dimension=2, size=self.kernel_size, step=1)
        # Median filtering to extract robust trend
        trend = x_unfolded.median(dim=-1)[0]
        trend = trend.permute(0, 2, 1) # [B, L, C]
        seasonal = x.permute(0, 2, 1) - trend
        return seasonal, trend

# --- 3. 核心改进模块 ---

class TimeBranch(nn.Module):
    """
    负责捕捉局部细节和高精度预测 (Fit for Clean Data)
    使用 MLP 结构
    """
    def __init__(self, input_len, pred_len, dropout=0.1):
        super(TimeBranch, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_len, pred_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len, pred_len)
        )
    
    def forward(self, x):
        # x: [B, C, L]
        return self.mlp(x)

class FreqBranch(nn.Module):
    """
    负责全局滤波和抗噪重构 (Fit for Noisy/Missing Data)
    使用 频域加权
    """
    def __init__(self, input_len, pred_len, enc_in, dropout=0.1):
        super(FreqBranch, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        
        # FFT 后的有效长度
        self.freq_len = input_len // 2 + 1
        
        # 学习复数权重：这是一个自适应的带通/低通滤波器
        self.complex_weight = nn.Parameter(
            torch.randn(enc_in, self.freq_len, 2, dtype=torch.float32) * 0.02
        )
        self.linear_map = nn.Linear(input_len, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, L]
        # 1. FFT
        x_fft = torch.fft.rfft(x, dim=-1)
        
        # 2. Filtering
        weight = torch.view_as_complex(self.complex_weight)
        x_fft = x_fft * weight
        
        # 3. iFFT
        x_ifft = torch.fft.irfft(x_fft, n=self.input_len, dim=-1)
        
        # 4. Mapping
        return self.dropout(self.linear_map(x_ifft))

class UncertaintyGate(nn.Module):
    """
    DUET 风格的门控融合：
    根据输入特征动态决定信赖 TimeBranch 还是 FreqBranch。
    如果噪声大，Gate 会偏向 FreqBranch；如果数据干净，偏向 TimeBranch。
    """
    def __init__(self, enc_in):
        super(UncertaintyGate, self).__init__()
        # 简单的 Squeeze-and-Excitation 风格门控
        self.gate_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), # Squeeze time dimension -> [B, C, 1]
            nn.Conv1d(enc_in, enc_in // 4 + 1, kernel_size=1), # Bottleneck
            nn.ReLU(),
            nn.Conv1d(enc_in // 4 + 1, enc_in, kernel_size=1), # Restore
            nn.Sigmoid() # 输出 0~1 之间的权重
        )

    def forward(self, x):
        # x: [B, C, L] -> weight: [B, C, 1]
        return self.gate_layer(x)

# --- 4. 主模型 ---
class Model(nn.Module):
    """
    Dual-Robust Mixer
    结合了 TimeMixer 的多尺度思想 + DUET 的双视图融合思想。
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # 1. RevIN (Drift)
        self.revin = RevIN(self.enc_in)
        
        # 2. Median Decomposition (Noise Robustness)
        # 确保 kernel 是奇数
        kernel_size = configs.moving_avg if configs.moving_avg % 2 == 1 else configs.moving_avg + 1
        self.decomposition = MedianDecomp(kernel_size)
        
        # 3. Trend Prediction (稳定基石)
        self.trend_linear = nn.Linear(self.seq_len, self.pred_len)
        self.trend_linear.weight = nn.Parameter(
            (1/self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # 4. Dual-View Seasonal Processing (核心)
        # Time Branch: 擅长细节 (无噪时占优)
        self.time_branch = TimeBranch(self.seq_len, self.pred_len, configs.dropout)
        
        # Freq Branch: 擅长去噪 (有噪/Dropout时占优)
        self.freq_branch = FreqBranch(self.seq_len, self.pred_len, self.enc_in, configs.dropout)
        
        # Gating Mechanism: 决定融合比例
        self.gate = UncertaintyGate(self.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, L, C]
        
        # Step 1: RevIN
        x = self.revin(x_enc, 'norm')

        # Step 2: Decomposition
        # 将原始数据拆解为 Trend (低频) 和 Seasonal (高频+噪声)
        # MedianDecomp 保证了 Trend 不会被高斯噪声带偏
        seasonal_init, trend_init = self.decomposition(x)
        
        # Step 3: Trend Prediction
        # [B, L, C] -> [B, C, L]
        trend_init = trend_init.permute(0, 2, 1)
        trend_output = self.trend_linear(trend_init)
        trend_output = trend_output.permute(0, 2, 1) # [B, Pred, C]

        # Step 4: Dual-View Seasonal Prediction (DUET Logic)
        seasonal_init = seasonal_init.permute(0, 2, 1) # [B, C, L]
        
        # A. 获取两种视图的预测
        out_time = self.time_branch(seasonal_init) # [B, C, Pred]
        out_freq = self.freq_branch(seasonal_init) # [B, C, Pred]
        
        # B. 计算门控权重
        # Gate 会根据 seasonal_init 的特征判断这是否是“难/脏”样本
        # alpha close to 1: Trust Time Branch (Clean data)
        # alpha close to 0: Trust Freq Branch (Noisy data)
        alpha = self.gate(seasonal_init) 
        
        # C. 加权融合
        seasonal_output = alpha * out_time + (1 - alpha) * out_freq
        seasonal_output = seasonal_output.permute(0, 2, 1) # [B, Pred, C]

        # Step 5: Final Sum & Denorm
        x = trend_output + seasonal_output
        x = self.revin(x, 'denorm')

        return x
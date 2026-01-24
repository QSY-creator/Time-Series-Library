import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. RevIN: 保持不变，它是处理 Distribution Drift 的最佳实践 ---
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

# --- 2. MedianDecomp: 保持中值分解，它对高斯噪声和尖峰噪声的鲁棒性优于均值 ---
class MedianDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(MedianDecomp, self).__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        x = x.permute(0, 2, 1) # [B, C, L]
        x_pad = F.pad(x, (self.padding, self.padding), mode='replicate')
        x_unfolded = x_pad.unfold(dimension=2, size=self.kernel_size, step=1)
        trend = x_unfolded.median(dim=-1)[0] # Robust Trend extraction
        trend = trend.permute(0, 2, 1) # [B, L, C]
        seasonal = x.permute(0, 2, 1) - trend
        return seasonal, trend

# --- 3. MixingBlock: 结合 TimeMixer 的多尺度处理和 TSMixer 的 MLP 滤波 ---
class MixingBlock(nn.Module):
    """
    负责处理单一尺度的数据。
    包含：独立的分解、线性的 Trend 预测、非线性的 Seasonal 滤波预测。
    """
    def __init__(self, input_len, pred_len, moving_avg, dropout):
        super(MixingBlock, self).__init__()
        
        # 独立的分解模块：不同尺度上的 Trend 定义是不同的
        self.decomposition = MedianDecomp(moving_avg)
        
        # Trend 分支: 简单的线性映射 (保持 DLinear 的稳定性)
        self.trend_layer = nn.Linear(input_len, pred_len)
        self.trend_layer.weight = nn.Parameter(
            (1/input_len) * torch.ones([pred_len, input_len])) # 初始化为平均值
        
        # Seasonal 分支: 升级为 MLP (借鉴 TSMixer/TiDE)
        # 相比单纯的 Linear，加入 Activation 和 Dropout 能更好地过滤高频噪声
        self.seasonal_layer = nn.Sequential(
            nn.Linear(input_len, pred_len),
            nn.GELU(),              # 非线性激活，帮助过滤低幅度的噪声干扰
            nn.Dropout(dropout),    # 这里的 Dropout 模拟训练时的随机性，增强鲁棒性
            nn.Linear(pred_len, pred_len)
        )

    def forward(self, x):
        # x: [Batch, Input_Len_Scale, Channel]
        
        # 1. 在当前尺度进行分解 (关键改进：在降噪后的视图上分解)
        seasonal, trend = self.decomposition(x)
        
        # 2. Trend 预测
        trend = trend.permute(0, 2, 1)   # [B, C, L]
        trend_out = self.trend_layer(trend) 
        trend_out = trend_out.permute(0, 2, 1) # [B, Pred, C]
        
        # 3. Seasonal 预测 (MLP Filter)
        seasonal = seasonal.permute(0, 2, 1)
        seasonal_out = self.seasonal_layer(seasonal)
        seasonal_out = seasonal_out.permute(0, 2, 1)
        
        # 4. 合并
        return trend_out + seasonal_out

# --- 4. 主模型: RobustTimeMixer ---
class Model(nn.Module):
    """
    改进版模型：
    1. 采用 TimeMixer 的 "Downsample-then-Decompose" 策略，
       利用 AvgPool 对抗 Dropout 和 Noise。
    2. Seasonal 部分引入 MLP 结构，增强去噪能力。
    3. RevIN 处理全局 Drift。
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dropout_rate = configs.dropout
        
        # 1. 归一化 (抗 Drift)
        self.revin = RevIN(self.enc_in)
        
        # 2. 多尺度设置
        # Scale 1: 原始分辨率，捕捉细节
        # Scale 2: 1/2 分辨率，平滑高斯噪声，填补 Dropout 空缺
        self.scales = [1, 2] 
        
        self.mixing_blocks = nn.ModuleList()
        self.down_sampling_layers = nn.ModuleList()
        
        for scale in self.scales:
            # 定义下采样
            if scale == 1:
                self.down_sampling_layers.append(nn.Identity())
                current_seq_len = self.seq_len
            else:
                # AvgPool 是天然的低通滤波器，对抗高斯噪声极佳
                self.down_sampling_layers.append(
                    nn.AvgPool1d(kernel_size=scale, stride=scale))
                current_seq_len = self.seq_len // scale
            
            # 为每个尺度创建一个独立的 MixingBlock
            # 窗口大小根据尺度调整，保证物理时间意义一致
            current_moving_avg = configs.moving_avg // scale if configs.moving_avg // scale > 1 else 3
            # 保证奇数 kernel size
            if current_moving_avg % 2 == 0: current_moving_avg += 1
            
            self.mixing_blocks.append(
                MixingBlock(current_seq_len, self.pred_len, current_moving_avg, self.dropout_rate)
            )
            
        # 3. 尺度融合权重 (可学习)
        # 允许模型自动学习在噪声大时更依赖 Scale 2，噪声小时利用 Scale 1
        self.scale_weight = nn.Parameter(torch.ones(len(self.scales)) / len(self.scales))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Seq_Len, Channel]
        
        # Step 1: RevIN Normalization (Global)
        x = self.revin(x_enc, 'norm')

        # Step 2: Multi-Scale Processing
        output_sum = torch.zeros([x.size(0), self.pred_len, x.size(2)], device=x.device)
        
        for i, scale in enumerate(self.scales):
            # A. 下采样输入 (Downsample Input)
            # 这与原代码不同。先下采样，实际上是在做去噪预处理。
            # 如果原始数据有 Random Dropout，AvgPool 会利用周围点补全信息。
            x_in = x.permute(0, 2, 1)
            x_down = self.down_sampling_layers[i](x_in)
            x_down = x_down.permute(0, 2, 1) # [B, L_down, C]
            
            # B. 独立的 Block 处理 (Decomp -> Linear/MLP)
            out = self.mixing_blocks[i](x_down)
            
            # C. 加权融合
            output_sum += out * self.scale_weight[i]

        # Step 3: RevIN Denormalization
        x_out = self.revin(output_sum, 'denorm')

        return x_out
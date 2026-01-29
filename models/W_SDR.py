import torch
import torch.nn as nn

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


class TokenMixer(nn.Module):
    """
    移植自 WPMixer: 负责在 Patch 维度（时间维度）上进行混合
    """
    def __init__(self, patch_num, expansion_factor, dropout):
        super(TokenMixer, self).__init__()
        self.patch_num = patch_num
        self.dropout = dropout
        
        self.layers = nn.Sequential(
            nn.Linear(self.patch_num, int(self.patch_num * expansion_factor)),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(self.patch_num * expansion_factor), self.patch_num)
        )

    def forward(self, x):
        # x: [Batch, Channel, d_model, Patch_Num]
        # Mix across Patch_Num dimension
        x = self.layers(x)
        return x


class ChannelMixer(nn.Module):
    """
    移植自 WPMixer: 负责在 Embedding 维度上进行混合
    """
    def __init__(self, d_model, expansion_factor, dropout):
        super(ChannelMixer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, int(d_model * expansion_factor)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * expansion_factor), d_model)
        )

    def forward(self, x):
        # x: [Batch, Channel, Patch_Num, d_model]
        x = self.layers(x)
        return x


class WPMixerBlock(nn.Module):
    """
    核心 Mixer 块：包含 TokenMixer 和 ChannelMixer
    """
    def __init__(self, patch_num, d_model, dropout=0.1, token_factor=5, channel_factor=5):
        super(WPMixerBlock, self).__init__()
        
        self.token_mixer = TokenMixer(patch_num, token_factor, dropout)
        self.channel_mixer = ChannelMixer(d_model, channel_factor, dropout)
        
        self.norm1 = nn.BatchNorm2d(d_model) # WPMixer use BN on d_model channel when transposed
        self.norm2 = nn.BatchNorm2d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input x: [Batch, Channel, Patch_Num, d_model]
        
        # 1. Token Mixing (Time/Patch mixing)
        # Norm expects [N, C, H, W], we treat d_model as Channels for BN
        # Permute to [Batch, d_model, Channel, Patch_Num] for BN and Linear
        y = x.permute(0, 3, 1, 2) 
        y = self.norm1(y)
        # Permute to [Batch, Channel, d_model, Patch_Num] for TokenMixer
        y = y.permute(0, 2, 1, 3)
        y = self.dropout(self.token_mixer(y))
        
        # Residual connection
        x = x + y.permute(0, 1, 3, 2) # Back to [Batch, Channel, Patch_Num, d_model]
        
        # 2. Channel Mixing (Embedding mixing)
        y = x.permute(0, 3, 1, 2) # [Batch, d_model, Channel, Patch_Num]
        y = self.norm2(y)
        y = y.permute(0, 2, 3, 1) # [Batch, Channel, Patch_Num, d_model]
        y = self.dropout(self.channel_mixer(y))
        
        # Residual connection
        x = x + y
        return x


class SeasonalPatchMixer(nn.Module):
    """
    替代原有的 SpectralLinear。
    使用 Patching + Mixer 的方式处理 Seasonal 高频分量。
    """
    def __init__(self, seq_len, pred_len, d_model, patch_len, stride, dropout=0.1):
        super(SeasonalPatchMixer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # 计算 Patch 数量
        self.patch_num = int((seq_len - patch_len) / stride + 1)
        # 处理 Padding 使得能够刚好切分 (简单处理：如果除不尽，在forward里补pad)
        # 这里为了稳健，我们使用 Unfold 自适应
        
        # Patch Embedding: Project patch_len -> d_model
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        
        # Mixer Layers (使用两层 Mixer 以增强非线性能力)
        self.mixer1 = WPMixerBlock(self.patch_num, d_model, dropout)
        self.mixer2 = WPMixerBlock(self.patch_num, d_model, dropout)
        
        # Final Projection
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1), # Flatten Patch_Num * d_model
            nn.Linear(self.patch_num * d_model, pred_len)
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        # Transpose to [Batch, Channel, Seq_Len] for unfolding
        x = x.permute(0, 2, 1)
        
        # 1. Patching
        # Unfold: [Batch, Channel, Patch_Num, Patch_Len]
        # Ensure we cover the sequence. If exact match isn't possible, we replicate last value
        pad_len = (self.patch_num - 1) * self.stride + self.patch_len - self.seq_len
        if pad_len > 0:
             x = torch.nn.functional.pad(x, (0, pad_len), mode='replicate')
             
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # 2. Embedding
        # [Batch, Channel, Patch_Num, Patch_Len] -> [Batch, Channel, Patch_Num, d_model]
        x_emb = self.patch_embedding(patches)
        x_emb = self.emb_dropout(x_emb)
        
        # 3. Mixing
        out = self.mixer1(x_emb)
        out = self.mixer2(out)
        
        # 4. Projection
        # [Batch, Channel, Patch_Num, d_model] -> [Batch, Channel, Pred_Len]
        out = self.head(out)
        
        # Back to [Batch, Pred_Len, Channel]
        return out.permute(0, 2, 1)


class Model(nn.Module):
    """
    Super_DP_RTM (Improved Version with WPMixer)
    
    结构：
    1. RevIN
    2. Decomposition (Moving Average)
    3. Trend Branch (Linear)
    4. Seasonal Branch (WPMixer-style Patching + Mixing)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # Config params extraction with defaults if not present
        d_model = getattr(configs, 'd_model', 128)
        dropout = getattr(configs, 'dropout', 0.1)
        patch_len = getattr(configs, 'patch_len', 16) # 默认 patch 长度
        stride = getattr(configs, 'stride', 8)        # 默认 stride
        
        # 1. RevIN
        self.revin = RevIN(self.enc_in)
        
        # 2. Series Decomposition
        self.decomposition = SeriesDecomp(25)
        
        # 3. Seasonal Branch: Replacing SpectralLinear with WPMixer-inspired Block
        # 确保 stride 和 patch_len 合理
        if self.seq_len < patch_len:
            patch_len = self.seq_len // 2
            stride = patch_len // 2
            
        self.seasonal_mixer = SeasonalPatchMixer(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            dropout=dropout
        )

        # 4. Trend Branch: Keep it simple (Linear) for robustness
        self.trend_projector = nn.Linear(self.seq_len, self.pred_len)
        self.trend_projector.weight.data.normal_(0, 1/self.seq_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Input_Len, Channel]
        
        # 1. RevIN Normalization
        x_enc = self.revin(x_enc, 'norm')

        # 2. Decomposition
        seasonal_init, trend_init = self.decomposition(x_enc)
        
        # 3. Dual Path Processing
        
        # Path 1: Seasonal (High Frequency) -> WPMixer Patching & Mixing
        # Captures local patterns and global dependencies via patch mixing
        seasonal_output = self.seasonal_mixer(seasonal_init)
        
        # Path 2: Trend (Low Frequency) -> Linear
        # Captures global trend
        trend_output = self.trend_projector(trend_init.permute(0, 2, 1)).permute(0, 2, 1)

        # 4. Summation
        x_out = seasonal_output + trend_output

        # 5. RevIN Denormalization
        x_out = self.revin(x_out, 'denorm')

        return x_out[:, -self.pred_len:, :]
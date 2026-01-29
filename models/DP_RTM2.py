import torch
import torch.nn as nn
import torch.nn.functional as F

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class MultiKernelDecomp(nn.Module):
    """
    D-PAD Inspired: Adaptive Multi-kernel Decomposition
    Captures trend from different frequency perspectives.
    """
    def __init__(self, kernel_sizes=[9, 17, 33]):
        super(MultiKernelDecomp, self).__init__()
        self.moving_avgs = nn.ModuleList([moving_avg(kernel, stride=1) for kernel in kernel_sizes])
        self.weights = nn.Parameter(torch.ones(len(kernel_sizes)) / len(kernel_sizes))

    def forward(self, x):
        trends = []
        for mv in self.moving_avgs:
            trends.append(mv(x))
        # Adaptive weighting for different kernel trends
        weights = F.softmax(self.weights, dim=0)
        trend = sum(w * t for w, t in zip(weights, trends))
        res = x - trend
        return res, trend

class GatedPatchMixer(nn.Module):
    """
    D-PAD Inspired: Dual-Path Patch Alignment Mixer with Gating
    Mixes information across Patches (Time) and Channels (Features).
    """
    def __init__(self, num_patches, d_model, c_in, dropout=0.1):
        super(GatedPatchMixer, self).__init__()
        # Temporal (Patch) Mixing
        self.time_mixer = nn.Sequential(
            nn.Linear(num_patches, num_patches * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_patches * 2, num_patches)
        )
        # Channel Mixing
        self.channel_mixer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.gate = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [Batch, Patches, d_model]
        
        # 1. Temporal (Patch) Mixing
        res = x
        x_time = self.norm1(x).transpose(1, 2) # [B, d_model, Patches]
        x_time = self.time_mixer(x_time)
        x_time = x_time.transpose(1, 2) # [B, Patches, d_model]
        x = res + x_time
        
        # 2. Channel Mixing with Gating (D-PAD alignment feature)
        res = x
        x_norm = self.norm2(x)
        x_chan = self.channel_mixer(x_norm)
        gate = torch.sigmoid(self.gate(x_norm)) # Gating mechanism for feature alignment
        x = res + x_chan * gate
        
        return x

class DisentangledBlock(nn.Module):
    """
    Processes Seasonal and Trend parts separately using Patch Mixers
    """
    def __init__(self, num_patches, d_model, c_in, dropout=0.1):
        super(DisentangledBlock, self).__init__()
        self.seasonal_mixer = GatedPatchMixer(num_patches, d_model, c_in, dropout)
        self.trend_mixer = GatedPatchMixer(num_patches, d_model, c_in, dropout)
        
    def forward(self, seasonal, trend):
        seasonal_out = self.seasonal_mixer(seasonal)
        trend_out = self.trend_mixer(trend)
        return seasonal_out, trend_out

class Model(nn.Module):
    """
    RTimeMixer2 Enhanced with D-PAD (Decoupling and Patching via Alignment)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_in = configs.enc_in
        self.d_model = getattr(configs, 'd_model', 128)
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)
        self.downsampling_layers = getattr(configs, 'down_sampling_layers', 2) # Inherited from TimeMixer

        # Calculate number of patches
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)
        
        # 1. Decomposition (D-PAD Core)
        self.decomp = MultiKernelDecomp(kernel_sizes=[9, 17, 33])
        
        # 2. Patch Embedding
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        
        # 3. Disentangled Mixers (Deep stack)
        self.layers = nn.ModuleList([
            DisentangledBlock(self.num_patches, self.d_model, self.c_in, configs.dropout)
            for _ in range(self.downsampling_layers)
        ])
        
        # 4. Independent Predictors for Trend and Seasonal
        self.seasonal_predictor = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.num_patches * self.d_model, self.pred_len)
        )
        
        self.trend_predictor = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.num_patches * self.d_model, self.pred_len)
        )
        
        # Residual connection projection if dimensions mismatch
        self.proj = nn.Linear(self.seq_len, self.pred_len)

    def _patching(self, x):
        # x: [Batch, Channel, Seq_len]
        B, C, L = x.shape
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [B, C, Num_Patches, Patch_Len]
        x = x.reshape(B * C, self.num_patches, self.patch_len) # [B*C, Num_Patches, Patch_Len]
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Input x_enc: [Batch, Seq_len, Channels]
        B, L, C = x_enc.shape
        
        # Normalization (Crucial for Non-stationarity)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # 1. Multi-Kernel Decomposition
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # Reshape for patching: [Batch, Channel, Seq_len]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        # 2. Patching and Embedding
        # [B*C, Num_Patches, Patch_Len]
        seasonal_patches = self._patching(seasonal_init)
        trend_patches = self._patching(trend_init)
        
        # [B*C, Num_Patches, d_model]
        seasonal_emb = self.patch_embedding(seasonal_patches)
        trend_emb = self.patch_embedding(trend_patches)
        
        # 3. Dual-Path Mixers
        for layer in self.layers:
            seasonal_emb, trend_emb = layer(seasonal_emb, trend_emb)
            
        # 4. Projection to Prediction Length
        # [B*C, Pred_len]
        seasonal_pred = self.seasonal_predictor(seasonal_emb)
        trend_pred = self.trend_predictor(trend_emb)
        
        # Reshape back: [Batch, Pred_len, Channel]
        seasonal_pred = seasonal_pred.reshape(B, C, self.pred_len).permute(0, 2, 1)
        trend_pred = trend_pred.reshape(B, C, self.pred_len).permute(0, 2, 1)
        
        # 5. Combine and De-normalize
        pred = seasonal_pred + trend_pred
        pred = pred * stdev + means
        
        return pred
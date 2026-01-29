import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN)
    Standard technique to handle non-stationarity in time series.
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
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
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.affine_weight.eq(0).float())
        x = x * self.stdev + self.mean
        return x

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels] -> Permute for AvgPool1d
        x = x.permute(0, 2, 1)
        # padding on the both ends of time series
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=2)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class MultiKernelDecomp(nn.Module):
    """
    Adaptive Multi-kernel Decomposition.
    Uses variable moving average windows to robustly extract trend.
    """
    def __init__(self, kernel_sizes=[9, 17, 33]):
        super(MultiKernelDecomp, self).__init__()
        self.moving_avgs = nn.ModuleList([moving_avg(kernel, stride=1) for kernel in kernel_sizes])
        self.weights = nn.Parameter(torch.ones(len(kernel_sizes)) / len(kernel_sizes))

    def forward(self, x):
        trends = []
        for mv in self.moving_avgs:
            trends.append(mv(x))
        
        # Adaptive weighting
        weights = F.softmax(self.weights, dim=0)
        trend = sum(w * t for w, t in zip(weights, trends))
        res = x - trend
        return res, trend

class GatedPatchMixer(nn.Module):
    """
    Enhanced Gated Mixer for Seasonal Components.
    Focuses on mixing information within patches and across channels efficiently.
    """
    def __init__(self, num_patches, d_model, dropout=0.1):
        super(GatedPatchMixer, self).__init__()
        
        # Temporal (Patch) Mixing: Communicates across time segments
        self.time_mixer = nn.Sequential(
            nn.Linear(num_patches, num_patches * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_patches * 2, num_patches)
        )
        
        # Feature (Embedding) Mixing: Communicates across hidden dims
        self.feature_mixer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # SuperLinear Gating: Controls information flow
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [Batch*Channel, Patches, d_model]
        
        # 1. Temporal Mixing (Transposed to mix patches)
        res = x
        x = self.norm1(x)
        x = x.transpose(1, 2) # [B*C, d_model, Patches]
        x = self.time_mixer(x)
        x = x.transpose(1, 2) # [B*C, Patches, d_model]
        x = res + self.dropout(x)
        
        # 2. Feature Mixing with Gating
        res = x
        x = self.norm2(x)
        gate_val = torch.sigmoid(self.gate(x))
        x = self.feature_mixer(x)
        x = res + self.dropout(x * gate_val)
        
        return x

class Model(nn.Module):
    """
    DP_RTM Enhanced with SuperLinear Concepts.
    
    Key Improvements:
    1. RevIN for robust normalization.
    2. Hybrid Architecture: 
       - Seasonal part uses Patching + Mixer (Local patterns).
       - Trend part uses Direct Linear Mapping (Global patterns, No patching).
    3. SuperLinear Shortcut: Direct projection from input to output.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_in = configs.enc_in
        self.d_model = getattr(configs, 'd_model', 128)
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)
        self.dropout = getattr(configs, 'dropout', 0.1)
        self.layers_num = getattr(configs, 'down_sampling_layers', 2)

        # 1. Normalization
        self.revin = RevIN(self.c_in, affine=True)

        # 2. Decomposition
        self.decomp = MultiKernelDecomp(kernel_sizes=[9, 17, 33])

        # ================= SEASONAL BRANCH (Patch + Mixer) =================
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        
        self.seasonal_layers = nn.ModuleList([
            GatedPatchMixer(self.num_patches, self.d_model, self.dropout)
            for _ in range(self.layers_num)
        ])
        
        # Head for Seasonal
        self.seasonal_head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.num_patches * self.d_model, self.pred_len),
            nn.Dropout(self.dropout)
        )

        # ================= TREND BRANCH (SuperLinear Style) =================
        # Trend is global, low-frequency. Patching it destroys information.
        # We use a single high-capacity Linear layer (or Multi-Layer Perceptron)
        self.trend_projector = nn.Linear(self.seq_len, self.pred_len)
        
        # ================= SUPERLINEAR SHORTCUT =================
        # A direct bypass to capture simplest AR relationships
        self.super_linear = nn.Linear(self.seq_len, self.pred_len)

    def _patching(self, x):
        # x: [Batch, Seq_len, Channel] -> [Batch * Channel, Num_Patches, Patch_Len]
        B, L, C = x.shape
        x = x.permute(0, 2, 1) # [B, C, L]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [B, C, N, P]
        x = x.reshape(B * C, self.num_patches, self.patch_len)
        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [Batch, Seq_len, Channels]
        B, L, C = x_enc.shape

        # 1. RevIN Normalization
        x_norm = self.revin(x_enc, 'norm')

        # 2. Decomposition
        # seasonal: [B, L, C], trend: [B, L, C]
        seasonal_part, trend_part = self.decomp(x_norm)

        # --- Branch 1: Trend Processing (SuperLinear Style) ---
        # Direct linear mapping for trend (efficient & accurate for low freq)
        # Permute to [B, C, L] for Linear layer
        trend_out = self.trend_projector(trend_part.permute(0, 2, 1)) # [B, C, Pred_len]
        trend_out = trend_out.permute(0, 2, 1) # [B, Pred_len, C]

        # --- Branch 2: Seasonal Processing (Patch Mixer Style) ---
        # Patching: [B*C, N, P]
        seasonal_patches = self._patching(seasonal_part)
        # Embedding: [B*C, N, d_model]
        x_seas = self.patch_embedding(seasonal_patches)
        
        # Deep Mixing
        for layer in self.seasonal_layers:
            x_seas = layer(x_seas)
            
        # Projection
        seasonal_out = self.seasonal_head(x_seas) # [B*C, Pred_len]
        seasonal_out = seasonal_out.reshape(B, C, self.pred_len).permute(0, 2, 1) # [B, Pred_len, C]

        # --- Branch 3: SuperLinear Shortcut ---
        # Captures global raw correlations
        raw_skip = self.super_linear(x_norm.permute(0, 2, 1)).permute(0, 2, 1)

        # 3. Summation
        # Combine: Trend + Seasonal + Shortcut
        final_pred = trend_out + seasonal_out + raw_skip

        # 4. RevIN Denormalization
        final_pred = self.revin(final_pred, 'denorm')

        return final_pred
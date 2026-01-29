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
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class MultiKernelDecomp(nn.Module):
    """
    S_DP_RTM Original Decomposition
    """
    def __init__(self, kernel_sizes=[9, 17, 33]):
        super(MultiKernelDecomp, self).__init__()
        self.moving_avgs = nn.ModuleList([moving_avg(kernel, stride=1) for kernel in kernel_sizes])
        self.weights = nn.Parameter(torch.ones(len(kernel_sizes)) / len(kernel_sizes))

    def forward(self, x):
        trends = []
        for mv in self.moving_avgs:
            trends.append(mv(x))
        weights = F.softmax(self.weights, dim=0)
        trend = sum(w * t for w, t in zip(weights, trends))
        res = x - trend
        return res, trend

class Model(nn.Module):
    """
    Optimized Hybrid Model:
    1. Backbone: Multi-Kernel Decomposition (From S_DP_RTM)
    2. Seasonal: GRU Encoder (From SegRNN) -> Global MLP Head (From S_DP_RTM)
       - GRU fixes Short-term (96,192) by modeling local dynamics/noise.
       - Global MLP fixes Long-term (336,720) by avoiding RNN memory bottleneck.
    3. Trend: Linear Interpolation (Robust for Long-term).
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = getattr(configs, 'd_model', 256) # 256 is usually enough
        self.dropout = getattr(configs, 'dropout', 0.1)
        
        # Patch/Seg Length:
        # Smaller patch (16) gives GRU more steps to reason about dynamics (Better for 96).
        # We stick to a standard size that balances both.
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)
        
        # Calculate number of patches
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)
        
        # --- 1. Decomposition ---
        self.decomp = MultiKernelDecomp(kernel_sizes=[9, 17, 33])
        
        # --- 2. Seasonal Model (GRU-Enhanced MLP) ---
        # Embedding
        self.sea_patch_embed = nn.Linear(self.patch_len, self.d_model)
        self.sea_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.d_model))
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Encoder: The SegRNN Soul (Captures Dynamics)
        self.sea_gru = nn.GRU(
            input_size=self.d_model, 
            hidden_size=self.d_model, 
            num_layers=1, 
            batch_first=True
        )
        
        # Head: The S_DP_RTM Soul (Global Context)
        # Instead of decoding step-by-step (SegRNN weakness), we flatten and project.
        self.sea_head = nn.Linear(self.num_patches * self.d_model, self.pred_len)

        # --- 3. Trend Model ---
        # For Trend, DLinear's simple mapping is unbeatable for stability on 720.
        # It avoids the overfitting of large Mixers on the trend component.
        self.trend_linear = nn.Linear(self.seq_len, self.pred_len)
        
    def _patching(self, x):
        # x: [B, C, L]
        B, C, L = x.shape
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.reshape(B * C, self.num_patches, self.patch_len)
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Seq_Len, Channels]
        B, L, C = x_enc.shape
        
        # === 1. RevIN ===
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # === 2. Decomposition ===
        # seasonal: [B, L, C], trend: [B, L, C]
        seasonal_part, trend_part = self.decomp(x_enc)
        
        # Permute for CI processing: [B, C, L]
        seasonal_part = seasonal_part.permute(0, 2, 1)
        trend_part = trend_part.permute(0, 2, 1)
        
        # === 3. Seasonal Processing (The Hybrid Core) ===
        # Patching: [B*C, N, P]
        sea_patches = self._patching(seasonal_part)
        
        # Embed: [B*C, N, D]
        sea_emb = self.sea_patch_embed(sea_patches)
        sea_emb = sea_emb + self.sea_pos_embed
        sea_emb = self.dropout_layer(sea_emb)
        
        # GRU Encoding (SegRNN Logic): 
        # Models the transitions between patches.
        # out: [B*C, N, D] -> Contains the full history context enriched by GRU
        sea_out, _ = self.sea_gru(sea_emb)
        
        # Global Projection (S_DP_RTM Logic):
        # Flatten: [B*C, N*D]
        sea_flat = sea_out.reshape(B * C, -1)
        
        # Projection: [B*C, Pred_Len]
        # This layer sees the WHOLE sequence history (via Flatten), 
        # solving the "bottleneck" issue of SegRNN on long horizons.
        seasonal_pred = self.sea_head(sea_flat)
        
        # Reshape: [B, C, Pred_Len]
        seasonal_pred = seasonal_pred.reshape(B, C, self.pred_len)
        
        # === 4. Trend Processing ===
        # Simple Linear Mapping: [B, C, L] -> [B, C, Pred_Len]
        # Robust against overfitting on long horizons.
        trend_pred = self.trend_linear(trend_part)
        
        # === 5. Sum & Re-permute ===
        pred = seasonal_pred + trend_pred
        pred = pred.permute(0, 2, 1) # [B, Pred_Len, C]
        
        # === 6. Denormalization ===
        pred = pred * stdev + means
        
        return pred
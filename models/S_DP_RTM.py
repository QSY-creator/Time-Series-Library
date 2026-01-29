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

class MLP_PatchMixer(nn.Module):
    """
    Standard MLP Mixer for Trend (Smooth, low frequency)
    """
    def __init__(self, num_patches, d_model, dropout=0.1):
        super(MLP_PatchMixer, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        # Global mixing for Trend
        self.mixer = nn.Sequential(
            nn.Linear(num_patches, num_patches),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.channel_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B*C, Num_Patches, d_model]
        res = x
        x = self.norm(x)
        
        # Mix over Time (Patches)
        x = x.transpose(1, 2) # [B*C, d_model, Num_Patches]
        x = self.mixer(x)
        x = x.transpose(1, 2) # [B*C, Num_Patches, d_model]
        
        # Mix over Features
        x = self.channel_proj(x)
        
        return res + self.dropout(x)

class SegRNN_Mixer(nn.Module):
    """
    SegRNN Inspired: Recurrent Mixer for Seasonal (High frequency, pattern evolution)
    Uses GRU to model the dependency between Patches (Segments).
    """
    def __init__(self, d_model, dropout=0.1):
        super(SegRNN_Mixer, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        
        # Core Improvement: GRU instead of MLP for patch interaction
        # Captures the sequential evolution of seasonality
        self.gru = nn.GRU(
            input_size=d_model, 
            hidden_size=d_model, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=False # Single direction often sufficient for causality, can be True
        )
        
        self.dropout = nn.Dropout(dropout)
        # Gating mechanism retained from DP_RTM logic, but applied to RNN output
        self.gate = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B*C, Num_Patches, d_model]
        res = x
        x = self.norm(x)
        
        # RNN Processing: Treat patches as a sequence steps
        # output: [B*C, Num_Patches, d_model]
        out, _ = self.gru(x)
        
        # Gating and Projection
        gate = torch.sigmoid(self.gate(out))
        out = self.proj(out) * gate
        
        return res + self.dropout(out)

class DisentangledBlock(nn.Module):
    """
    Hybrid Processing Block:
    - Seasonal: Processed by SegRNN (Recurrent) to capture pattern shifts.
    - Trend: Processed by MLP (Global) to capture overall trajectory.
    """
    def __init__(self, num_patches, d_model, dropout=0.1):
        super(DisentangledBlock, self).__init__()
        # Seasonal gets the RNN treatment (SegRNN)
        self.seasonal_mixer = SegRNN_Mixer(d_model, dropout)
        # Trend gets the MLP treatment (Simpler, smoother)
        self.trend_mixer = MLP_PatchMixer(num_patches, d_model, dropout)
        
    def forward(self, seasonal, trend):
        seasonal_out = self.seasonal_mixer(seasonal)
        trend_out = self.trend_mixer(trend)
        return seasonal_out, trend_out

class Model(nn.Module):
    """
    DP_RTM Enhanced with SegRNN Concepts.
    Replaces purely MLP mixing with Segment-wise Recurrent mixing for seasonal components.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_in = configs.enc_in
        self.d_model = getattr(configs, 'd_model', 256) # SegRNN usually benefits from slightly larger dim
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)
        self.dropout = getattr(configs, 'dropout', 0.1)
        
        # SegRNN typically uses fewer layers than Transformers
        self.num_layers = getattr(configs, 'e_layers', 1) 

        # Calculate number of patches
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)
        
        # 1. Decomposition
        self.decomp = MultiKernelDecomp(kernel_sizes=[9, 17, 33])
        
        # 2. Patch Embedding (Linear Segment Projection)
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.d_model))
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # 3. Hybrid Encoders
        self.layers = nn.ModuleList([
            DisentangledBlock(self.num_patches, self.d_model, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # 4. Predictors
        # Flattening [Num_Patches * d_model] -> [pred_len]
        self.head_nf = self.d_model * self.num_patches
        self.seasonal_head = nn.Linear(self.head_nf, self.pred_len)
        self.trend_head = nn.Linear(self.head_nf, self.pred_len)

    def _patching(self, x):
        # x: [Batch, Channel, Seq_len]
        B, C, L = x.shape
        # Unfold to get patches
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [B, C, Num_Patches, Patch_Len]
        x = x.reshape(B * C, self.num_patches, self.patch_len) # [B*C, Num_Patches, Patch_Len]
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Input x_enc: [Batch, Seq_len, Channels]
        B, L, C = x_enc.shape
        
        # --- 1. RevIN Normalization ---
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # --- 2. Decomposition ---
        # Decompose raw series before embedding
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # Permute for patching: [B, C, L]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        # --- 3. Patching & Embedding ---
        # [B*C, Num_Patches, Patch_Len]
        seasonal_patches = self._patching(seasonal_init)
        trend_patches = self._patching(trend_init)
        
        # [B*C, Num_Patches, d_model]
        seasonal_emb = self.patch_embedding(seasonal_patches)
        trend_emb = self.patch_embedding(trend_patches)
        
        # Add Positional Encoding (Crucial for the Transformer/RNN hybrid feel)
        seasonal_emb = seasonal_emb + self.pos_embedding
        trend_emb = trend_emb + self.pos_embedding
        
        seasonal_emb = self.dropout_layer(seasonal_emb)
        trend_emb = self.dropout_layer(trend_emb)
        
        # --- 4. Hybrid Processing (SegRNN Logic + MLP Logic) ---
        for layer in self.layers:
            seasonal_emb, trend_emb = layer(seasonal_emb, trend_emb)
            
        # --- 5. Prediction Head ---
        # Flatten: [B*C, Num_Patches * d_model]
        seasonal_flat = seasonal_emb.reshape(B * C, -1)
        trend_flat = trend_emb.reshape(B * C, -1)
        
        seasonal_pred = self.seasonal_head(seasonal_flat)
        trend_pred = self.trend_head(trend_flat)
        
        # Reshape back: [Batch, Pred_len, Channel]
        seasonal_pred = seasonal_pred.reshape(B, C, self.pred_len).permute(0, 2, 1)
        trend_pred = trend_pred.reshape(B, C, self.pred_len).permute(0, 2, 1)
        
        # --- 6. Denormalization ---
        pred = seasonal_pred + trend_pred
        pred = pred * stdev + means
        
        return pred
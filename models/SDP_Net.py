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
    Adaptive Multi-kernel Decomposition from S_DP_RTM
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

class Model(nn.Module):
    """
    Hybrid Model: SDP-Net (Seg-Decomposed Parallel Network)
    
    Strategy:
    1. Decompose Trend and Seasonal (Like S_DP_RTM).
    2. Seasonal Branch: Processed by SegRNN style Encoder-Decoder (GRU + PMF). 
       This fixes the performance on short horizons (96, 192).
    3. Trend Branch: Processed by a simplified MLP Mixer.
       This fixes the performance on long horizons (336, 720).
    4. Channel Independence is strictly enforced.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = getattr(configs, 'd_model', 512)
        self.dropout = getattr(configs, 'dropout', 0.1)
        
        # Patch/Seg Length configuration
        # SegRNN prefers 48, Transformer/MLP usually 16. 
        # We choose 24 or 48 as a sweet spot for hybrid. Defaulting to 48 (SegRNN style) if not specified.
        self.patch_len = getattr(configs, 'patch_len', 48) # Also acts as seg_len
        
        # Calculate number of patches for Input (Encoder) and Output (Decoder)
        # Note: We assume padding is handled in forward to make division exact
        self.enc_patch_num = (self.seq_len + self.patch_len - 1) // self.patch_len
        self.dec_patch_num = (self.pred_len + self.patch_len - 1) // self.patch_len
        
        # --- 1. Decomposition ---
        self.decomp = MultiKernelDecomp(kernel_sizes=[9, 17, 33])
        
        # --- 2. Seasonal Branch (SegRNN Architecture) ---
        # Encoder: Projects patch -> d_model
        self.sea_enc_proj = nn.Linear(self.patch_len, self.d_model)
        # GRU Encoder: Captures sequential dependencies of seasonal patterns
        self.sea_gru_enc = nn.GRU(
            input_size=self.d_model, hidden_size=self.d_model, 
            num_layers=1, batch_first=True
        )
        # Decoder: Parallel Multi-step Forecasting (PMF)
        # Learnable Positional Embeddings for the future segments
        self.sea_pos_emb = nn.Parameter(torch.randn(self.dec_patch_num, self.d_model))
        # GRU Decoder: Takes history state + pos_emb to predict future
        self.sea_gru_dec = nn.GRU(
            input_size=self.d_model, hidden_size=self.d_model,
            num_layers=1, batch_first=True
        )
        self.sea_out_proj = nn.Linear(self.d_model, self.patch_len)
        
        # --- 3. Trend Branch (MLP/Global Architecture) ---
        # Trend is low frequency, so a simple global MLP works best for long horizons (336, 720)
        # We use a linear mapping across time to project the trend.
        self.trend_model = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Seq_Len, Channels]
        B, L, C = x_enc.shape
        
        # === 1. RevIN Normalization ===
        # Standard RevIN stats
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # === 2. Channel Independence Transformation ===
        # Merge Batch and Channel: [B*C, L, 1]
        x_enc = x_enc.permute(0, 2, 1).reshape(B * C, L, 1)
        
        # === 3. Decomposition ===
        # Apply decomp on the flattened CI data
        seasonal, trend = self.decomp(x_enc) # [B*C, L, 1]
        
        # Remove the last dim for processing: [B*C, L]
        seasonal = seasonal.squeeze(-1)
        trend = trend.squeeze(-1)

        # === 4. Seasonal Processing (SegRNN Style) ===
        # A. Padding if necessary
        if L % self.patch_len != 0:
            pad_len = self.patch_len - (L % self.patch_len)
            seasonal = torch.cat([seasonal, torch.zeros(B*C, pad_len).to(seasonal.device)], dim=1)
        
        # B. Patching: [B*C, Num_Patches, Patch_Len]
        num_patches = seasonal.shape[1] // self.patch_len
        seasonal_patches = seasonal.reshape(B * C, num_patches, self.patch_len)
        
        # C. Encoding
        # [B*C, Num_Patches, d_model]
        enc_out = self.sea_enc_proj(seasonal_patches)
        enc_out = self.dropout_layer(F.gelu(enc_out))
        
        # GRU Pass: We only need the final hidden state for the Decoder initialization
        # _, h_n = self.sea_gru_enc(enc_out) -> h_n: [1, B*C, d_model]
        _, h_n = self.sea_gru_enc(enc_out)
        
        # D. Decoding (PMF Strategy)
        # Expand Positional Embeddings for batch: [B*C, Dec_Num_Patches, d_model]
        dec_in = self.sea_pos_emb.unsqueeze(0).repeat(B * C, 1, 1)
        
        # Run GRU Decoder using Encoder's final state as init
        dec_out, _ = self.sea_gru_dec(dec_in, h_n)
        
        # Project back to time domain: [B*C, Dec_Num_Patches, Patch_Len]
        seasonal_pred = self.sea_out_proj(dec_out)
        
        # Flatten and crop to pred_len: [B*C, Pred_Len]
        seasonal_pred = seasonal_pred.reshape(B * C, -1)
        seasonal_pred = seasonal_pred[:, :self.pred_len]
        
        # === 5. Trend Processing (Global MLP Style) ===
        # Trend branch takes the full trend sequence and maps to pred sequence
        # [B*C, L] -> [B*C, Pred_Len]
        trend_pred = self.trend_model(trend)
        
        # === 6. Summation & Reshape ===
        pred = seasonal_pred + trend_pred
        
        # Reshape back to [B, C, Pred_Len] -> [B, Pred_Len, C]
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)
        
        # === 7. Denormalization ===
        pred = pred * stdev + means
        
        return pred
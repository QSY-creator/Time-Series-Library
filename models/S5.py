import torch
import torch.nn as nn

class Model(nn.Module):
    """
    DeRNN: Decomposed Recurrent Neural Network
    Structure: 
        - Track A (Anchor): Global Trend (Linear + Dropout)
        - Track B (Feature): Local Dynamics (Patch-GRU)
        - Fusion: Adaptive Gated Fusion
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        
        # Segment length for Patching (SegRNN concept)
        self.seg_len = getattr(configs, 'seg_len', 24)
        
        # ============================================================
        # Track 1: Anchor Track (Global Trend)
        # ------------------------------------------------------------
        # Applies a linear mapping across the entire look-back window.
        # [NEW]: Added Dropout to prevent overfitting on small datasets (ETTh1).
        # ============================================================
        self.anchor_dropout = nn.Dropout(self.dropout)
        self.anchor_linear = nn.Linear(self.seq_len, self.pred_len)
        
        # ============================================================
        # Track 2: Feature Track (Local Dynamics)
        # ------------------------------------------------------------
        # Captures local volatility and details using Patch-wise GRU.
        # ============================================================
        self.feature_embedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.gru = nn.GRU(
            input_size=self.d_model, 
            hidden_size=self.d_model, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.decode_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.d_model), # *2 for bidirectional
            nn.ReLU(),
            nn.Linear(self.d_model, self.pred_len)
        )
        
        # ============================================================
        # Adaptive Fusion Gate
        # ------------------------------------------------------------
        # Learnable weights to balance Trend vs. Details
        # ============================================================
        self.combine_weight = nn.Parameter(torch.ones(2))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc shape: [Batch, Seq_Len, Channels]
        B, L, C = x_enc.shape
        
        # -----------------------------------------------
        # Step 1: RevIN (Normalization) - 稳健性基石
        # -----------------------------------------------
        means = x_enc.mean(1, keepdim=True)
        x_enc_norm = x_enc - means
        stds = torch.sqrt(torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_norm /= stds

        # -----------------------------------------------
        # Track A: Anchor Track (Trend Processing)
        # -----------------------------------------------
        # Permute to [Batch, Channels, Seq_Len] so Linear applies to time dimension
        # resulting in Channel Independence (CI).
        trend_input = x_enc_norm.permute(0, 2, 1) 
        
        # [关键修改]: 先 Dropout 再 Linear，强迫模型学习鲁棒特征
        trend_input = self.anchor_dropout(trend_input)
        
        trend_pred = self.anchor_linear(trend_input)  # [Batch, Channels, Pred_Len]
        trend_pred = trend_pred.permute(0, 2, 1)      # [Batch, Pred_Len, Channels]

        # -----------------------------------------------
        # Track B: Feature Track (Detail Processing)
        # -----------------------------------------------
        # 1. Reshape for Channel Independence: [B*C, L, 1]
        x_local = x_enc_norm.permute(0, 2, 1).reshape(B * C, L, 1)
        
        # 2. Patching / Segmentation
        # Handle padding if L is not divisible by seg_len
        if L % self.seg_len != 0:
            pad_len = self.seg_len - (L % self.seg_len)
            # Replicate padding is safer than zero padding for time series
            last_val = x_local[:, -1:, :]
            x_local = torch.cat([x_local, last_val.repeat(1, pad_len, 1)], dim=1)
        
        # 3. Embedding: [B*C, Num_Patches, Seg_Len] -> [B*C, Num_Patches, D_Model]
        x_local = x_local.reshape(B * C, -1, self.seg_len)
        x_local = self.feature_embedding(x_local)
        
        # 4. GRU Encoding
        _, h_n = self.gru(x_local) # h_n: [2, B*C, D_Model] (2 for bidirectional)
        
        # Flatten bidirectional outputs
        h_n = h_n.permute(1, 0, 2).reshape(B * C, -1) # [B*C, 2*D_Model]
        
        # 5. Projection (Decoding)
        detail_pred = self.decode_proj(h_n) # [B*C, Pred_Len]
        detail_pred = detail_pred.reshape(B, C, self.pred_len).permute(0, 2, 1) # [B, Pred_Len, C]

        # -----------------------------------------------
        # Step 3: Adaptive Fusion & Denormalization
        # -----------------------------------------------
        # Softmax ensures weights sum to 1
        w = torch.softmax(self.combine_weight, dim=0)
        
        # Fusion: Weighted sum of Trend and Detail
        final_pred = w[0] * trend_pred + w[1] * detail_pred
        
        # RevIN Inverse (Denormalization)
        final_pred = final_pred * stds + means
        
        return final_pred
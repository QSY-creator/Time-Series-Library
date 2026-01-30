import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
        return x

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        # x: [B, L, C]
        moving_mean = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        
        # 1. 引入 RevIN
        self.revin = RevIN(configs.enc_in)
        
        # 2. 引入 趋势-周期 分解
        self.decomp = SeriesDecomp(kernel_size=25)
        
        # 3. 多尺度分段: 使用两个不同的 patch_len
        self.seg_lens = [24, 48] 
        self.d_seg = self.d_model // len(self.seg_lens)
        
        # 编码层：每个尺度独立的线性映射
        self.value_embeddings = nn.ModuleList([
            nn.Linear(slen, self.d_seg) for slen in self.seg_lens
        ])
        
        # 核心 RNN
        self.rnn = nn.GRU(input_size=self.d_seg * len(self.seg_lens), 
                          hidden_size=self.d_model, 
                          num_layers=1, 
                          batch_first=True)
        
        # 4. 增强型解码器
        self.pos_emb = nn.Parameter(torch.randn(self.pred_len // 24 + 1, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
        
        self.predict = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.d_model, 24) # 假设基础解码单元长度为 24
        )

    def _process_segment(self, x, seg_len, embed_layer):
        # x: [BC, L, 1]
        B, L, _ = x.shape
        if L % seg_len != 0:
            pad_len = seg_len - (L % seg_len)
            x = torch.cat([x, x[:, -pad_len:, :]], dim=1)
        
        num_seg = x.shape[1] // seg_len
        x = x.reshape(B, num_seg, seg_len)
        return embed_layer(x) # [BC, num_seg, d_seg]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, C = x_enc.shape
        
        # === 1. Normalization ===
        x_enc = self.revin(x_enc, 'norm')
        
        # === 2. Decomposition ===
        seasonal, trend = self.decomp(x_enc)
        
        # === 3. Encoding (Channel Independence) ===
        seasonal = seasonal.permute(0, 2, 1).contiguous().reshape(B * C, L, 1)
        
        # 多尺度特征提取并拼接
        features = []
        for i, slen in enumerate(self.seg_lens):
            feat = self._process_segment(seasonal, slen, self.value_embeddings[i])
            # 对齐尺度，取最后一个时间步的特征或插值
            features.append(feat[:, -1:, :]) 
        
        enc_out = torch.cat(features, dim=-1) # [BC, 1, d_model]
        
        # GRU 编码
        _, hn = self.rnn(enc_out) # hn: [1, BC, d_model]
        
        # === 4. Decoding (With Channel & Positional Info) ===
        # 构造解码查询向量
        num_dec_steps = self.pred_len // 24 + (1 if self.pred_len % 24 != 0 else 0)
        
        # [C, num_dec_steps, d_model/2]
        p_emb = self.pos_emb[:num_dec_steps, :].unsqueeze(0).repeat(C, 1, 1)
        c_emb = self.channel_emb.unsqueeze(1).repeat(1, num_dec_steps, 1)
        dec_query = torch.cat([p_emb, c_emb], dim=-1) # [C, M, d_model]
        dec_query = dec_query.repeat(B, 1, 1) # [BC, M, d_model]
        
        # 解码 GRU
        dec_out, _ = self.rnn(dec_query, hn)
        
        # 映射回时间序列
        res = self.predict(dec_out) # [BC, M, 24]
        res = res.reshape(B, C, -1)[:, :, :self.pred_len].permute(0, 2, 1)
        
        # === 5. Trend Handling (Linear Projection for Trend) ===
        # 趋势部分通常较平滑，用一个简单的线性层效果最好
        trend_out = self._predict_trend(trend)
        
        # 合并并反归一化
        out = res + trend_out
        return self.revin(out, 'denorm')

    def _predict_trend(self, trend):
        # 极简趋势预测：线性外推
        # trend: [B, L, C]
        B, L, C = trend.shape
        trend_flat = trend.permute(0, 2, 1).reshape(B * C, L)
        # 实际上这里可以换成更复杂的 Linear 层
        # 为了演示，我们简单取最后一段的均值或线性映射
        trend_lin = nn.Linear(L, self.pred_len).to(trend.device)
        out = trend_lin(trend_flat).reshape(B, C, self.pred_len).permute(0, 2, 1)
        return out
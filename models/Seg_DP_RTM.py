import torch
import torch.nn as nn

class Model(nn.Module):
    """
    改进版 DP_RTM (Based on SegRNN Architecture)
    
    主要改进：
    1. 引入 Patching (分段) 机制，将 RNN 迭代次数减少 (L/patch_len) 倍。
    2. 采用 Parallel Multi-step Forecasting (PMF)，消除解码累积误差。
    3. Channel Independence (CI) 策略，提高泛化能力。
    
    参数建议 (可在 run.py 或 命令行中调整):
    - d_model: 推荐 512
    - dropout: 推荐 0.5
    - seg_len (patch_len): 推荐 48 (对于 seq_len 720/512/336)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # 基础配置
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        
        # 这里的 seg_len 对应 SegRNN 中的 patch size
        # 如果 configs 里没有定义 seg_len，则默认取 48 (SegRNN 论文常用值)
        self.seg_len = getattr(configs, 'seg_len', 48)
        
        # 确保序列长度能被分段长度整除，简单起见这里做padding处理的逻辑在forward里
        self.enc_seg_num = self.seq_len // self.seg_len
        self.dec_seg_num = self.pred_len // self.seg_len
        
        # 如果不能整除，需要微调 seg_len 或在 forward 中 padding，
        # 为了稳定性，建议 seq_len 和 pred_len 最好是 12 或 24 的倍数。
        if self.seq_len % self.seg_len != 0:
            # 简单的自动调整策略，或者打印警告
            pass 

        # --- 核心模块 ---
        
        # 1. 线性投影: 将分段后的 (Batch, Seg_Num, Seg_Len) 映射到 (Batch, Seg_Num, d_model)
        self.input_projection = nn.Linear(self.seg_len, self.d_model)
        
        # 2. 编码器 GRU: 处理 Segment 序列
        # 注意：这里 batch_first=True
        self.gru = nn.GRU(input_size=self.d_model, 
                          hidden_size=self.d_model, 
                          num_layers=1, 
                          batch_first=True)
        
        # 3. 解码器位置编码 (Positional Embedding for Prediction)
        # 为预测的每一段学习一个位置向量
        self.pos_emb = nn.Parameter(torch.randn(self.dec_seg_num, self.d_model))
        
        # 4. 通道位置编码 (可选，SegRNN 论文中提到这对多变量有帮助)
        # 考虑到 Channel Independence，我们通常让模型学习 Embedding
        # 这里简化为纯 CI 策略，不强制加 Channel Embedding 以保证通用性
        
        # 5. 输出投影: 将 (Batch, Dec_Seg_Num, d_model) 映射回 (Batch, Dec_Seg_Num, Seg_Len)
        self.output_projection = nn.Linear(self.d_model, self.seg_len)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.activation = nn.ReLU()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        x_enc: (Batch, Seq_Len, Channels)
        """
        # === 1. Normalization (RevIN 风格，即插即用) ===
        # 减去序列最后一个值，这是 SegRNN 论文中的简单归一化技巧
        seq_last = x_enc[:, -1:, :].detach()
        x_enc = x_enc - seq_last

        # === 2. Channel Independence ===
        # 将 (Batch, Seq_Len, Channels) -> (Batch * Channels, Seq_Len, 1)
        B, L, C = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * C, L, 1)
        
        # === 3. Segmentation (Patching) ===
        # 目标: (Batch * Channels, Seg_Num, Seg_Len)
        # 如果长度不够整除，简单的 padding 策略
        if L % self.seg_len != 0:
            pad_len = self.seg_len - (L % self.seg_len)
            x_enc = torch.cat([x_enc, torch.zeros(B*C, pad_len, 1).to(x_enc.device)], dim=1)
            
        num_seg = x_enc.shape[1] // self.seg_len
        x_enc = x_enc.reshape(B * C, num_seg, self.seg_len)

        # === 4. Encoding ===
        # Projection: (BC, N, S) -> (BC, N, D)
        enc_out = self.input_projection(x_enc)
        enc_out = self.activation(enc_out)
        enc_out = self.dropout_layer(enc_out)
        
        # GRU Encoding: 输出 (output, h_n)
        # output: (BC, N, D), h_n: (1, BC, D)
        _, h_n = self.gru(enc_out)
        
        # h_n 包含了历史序列的压缩信息
        # h_n shape: (1, BC, D) -> squeeze -> (BC, D)
        h_n = h_n.squeeze(0) 

        # === 5. Decoding (PMF Strategy) ===
        # 我们需要预测 P 个段。SegRNN 的做法是并行解码。
        # 核心思想：Predict_Segment[i] = GRU_Cell(Pos_Emb[i], h_n) 
        # 但 SegRNN 官方实现中，其实是将 h_n 扩展后与 Pos_Emb 结合，再过一次 MLP 或 GRU。
        # 这里为了最大化利用 GRU 参数，我们采用如下策略：
        # 将 h_n 视为初始状态，输入是 Position Embeddings
        
        # 准备解码输入: (BC, Pred_Seg_Num, D)
        # 扩展 Position Embedding 到 Batch 大小
        dec_in = self.pos_emb.unsqueeze(0).repeat(B * C, 1, 1) # (BC, M, D)
        
        # 在 SegRNN 中，解码器其实也是用 GRU 处理。
        # 这里我们将 Encoder 的最终状态 h_n 作为 Decoder GRU 的初始 hidden state。
        # 输入是位置编码。这样 GRU 会基于位置编码和历史状态流出信息。
        
        # h_n: (1, BC, D)
        h_n_dec = h_n.unsqueeze(0) 
        
        # 运行解码 GRU
        dec_out, _ = self.gru(dec_in, h_n_dec) # dec_out: (BC, M, D)
        
        # === 6. Output Projection ===
        dec_out = self.dropout_layer(dec_out)
        # (BC, M, D) -> (BC, M, Seg_Len)
        pred_out = self.output_projection(dec_out)
        
        # === 7. Reshape & Denormalization ===
        # Reshape 回 (Batch, Channels, Pred_Len)
        pred_out = pred_out.reshape(B, C, -1)
        
        # 截取需要的长度 (如果 Pred_Len 不能被 Seg_Len 整除)
        pred_out = pred_out[:, :, :self.pred_len]
        
        # 变回 (Batch, Pred_Len, Channels)
        pred_out = pred_out.permute(0, 2, 1)
        
        # 反归一化 (+ seq_last)
        pred_out = pred_out + seq_last

        return pred_out
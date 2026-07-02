"""
SSSS: Sparse, Simple, Stable, Scalable - 双分支多变量长期时间序列预测模型
================================================================================

架构概述:
    SSSS 采用双分支并行架构，将目标变量与特征变量的预测解耦处理：
    1. Anchor Track (目标变量专用通道):
       - 直接对目标变量序列 (最后一列) 做线性变换，学习自相关性
       - 简单有效，避免特征变量噪声干扰目标变量的预测

    2. Feature Track (特征变量抗噪通道):
       - 对所有特征（含目标变量）进行归一化 → Patching → Embedding → GRU编码
       - 采用 Direct Projection 解码：一次性将 GRU 隐状态映射到完整预测长度
       - 避开自回归递归，解决长序列预测（如 720 步）中误差累积问题
       - 通道独立 (Channel Independence, CI) 策略：将多变量序列独立处理

    最终融合:
       - 目标变量：两个分支的结果通过可学习权重加权融合
       - 其他特征变量：直接使用 Feature Track 的输出

核心设计思想:
    - 解耦设计: 目标变量和特征变量分开处理，互不干扰
    - 抗噪能力: Feature Track 通过 Embedding + Dropout 过滤特征噪声
    - 长序列鲁棒性: Direct Projection 避免递归式解码在长预测窗口下的退化
    - 轻量化: 仅使用 Linear + GRU，参数量小，训练稳定快速

输入/输出:
    输入:  x_enc  [B, L, C]  - 编码器输入序列 (B: batch, L: lookback窗口长度, C: 变量数)
    输出:           [B, pred_len, C]  - 预测序列
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    SSSS 模型主体

    Args:
        configs: 配置对象，包含以下关键属性：
            - seq_len (int): 输入序列长度 (回看窗口大小, lookback length)
            - pred_len (int): 预测序列长度 (forecast horizon)，支持长达 720 步
            - enc_in (int): 输入变量数 (特征通道数 C)
            - d_model (int): 隐藏层维度 (模型容量控制)
            - seg_len (int): patching 的片段长度，默认 24，控制局部模式粒度
            - dropout (float): Dropout 比率，用于正则化和抗噪
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # ==================== 基础配置 ====================
        self.seq_len = configs.seq_len        # 输入回看窗口长度 L
        self.pred_len = configs.pred_len      # 预测长度 H
        self.enc_in = configs.enc_in          # 变量通道数 C
        self.d_model = configs.d_model        # 模型隐藏层维度

        # seg_len: patching 的片段长度，控制将长序列切分为多短的片段
        # 较小的 seg_len 捕获细粒度局部模式，较大的捕获粗粒度趋势
        self.seg_len = getattr(configs, 'seg_len', 24)

        # ==================== 分支一: 目标变量专用通道 (Anchor Track) ====================
        # 动机: 目标变量本身包含着最强的自相关信号，用简单的 Linear 层
        # 直接从历史序列映射到未来序列，避免特征变量带来额外噪声。
        # 这是一个纯粹的时序外推：seq_len 个历史点 → pred_len 个未来点
        self.target_linear = nn.Linear(self.seq_len, self.pred_len)

        # ==================== 分支二: 特征变量抗噪通道 (Feature Track) ====================
        # 第一步: 特征嵌入与噪声过滤
        #   - Linear(seg_len, d_model): 将每个 patch 片段映射到高维空间
        #   - ReLU: 非线性激活，增强表达能力
        #   - Dropout: 随机丢弃，防止过拟合，同时起到噪声过滤作用
        self.feature_embedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )

        # 第二步: 时序编码 (GRU)
        #   使用双向 GRU (bidirectional=True) 来捕获序列的全局依赖关系:
        #   - input_size=d_model: 每个 patch 嵌入后的维度
        #   - hidden_size=d_model: GRU 隐状态维度
        #   - num_layers=1: 单层 GRU，保持轻量
        #   - batch_first=True: 输入/输出维度格式为 [B, T, D] (batch, time, feature)
        #   - bidirectional=True: 双向扫描，输出隐状态维度变为 2*d_model
        #   GRU 能有效捕获长序列（如 720 步）的全局时序模式
        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 第三步: 直接投影解码 (Direct Projection)
        #   将 GRU 的最终隐状态（双向拼接后维度 2*d_model）直接映射到完整的预测序列
        #   解码结构: 2*d_model → d_model → pred_len
        #   关键优势: 不采用逐步自回归（autoregressive）方式，
        #   避免了递归解码在长预测窗口（如 720 步）下的误差累积和性能退化
        self.decode_proj = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),  # 2*d_model → d_model (双向→单维度压缩)
            nn.ReLU(),
            nn.Linear(self.d_model, self.pred_len)       # d_model → pred_len (一步出完整预测)
        )

        # ==================== 融合权重 (可学习) ====================
        # 两个可学习参数，经过 softmax 后加权融合两个分支的目标变量预测
        # 初始化为 [1, 1]，softmax 后初始权重各为 0.5
        # 训练过程中模型会自动学习最优的融合比例
        self.combine_weight = nn.Parameter(torch.ones(2))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        前向传播

        Args:
            x_enc [B, L, C]: 编码器输入序列
                B - batch size
                L - 输入序列长度 (lookback window = seq_len)
                C - 变量/通道数 (含目标变量在内的所有特征列)
            x_mark_enc: 编码器时间戳标记 (本模型未使用)
            x_dec: 解码器输入 (本模型未使用，SSSS 不依赖 decoder 输入)
            x_mark_dec: 解码器时间戳标记 (本模型未使用)
            mask: 掩码 (本模型未使用)

        Returns:
            out [B, pred_len, C]: 预测序列，其中:
                - 最后一列 (目标变量) 由双分支加权融合得到
                - 其他列直接来自 Feature Track 的输出
        """

        # ==================== 获取输入形状 ====================
        B, L, C = x_enc.shape
        # B: batch size
        # L: lookback 窗口长度 (= seq_len)
        # C: 变量通道数 (= enc_in)，最后一列为目标预测变量

        # ==================== 分支一: 纯净目标变量轨迹 (Anchor Track) ====================
        # 步骤: 提取目标列 → Linear 映射 → 恢复通道维度

        # 取出目标变量序列（默认假设最后一列为待预测变量）
        target = x_enc[:, :, -1]  # [B, L]  - 仅保留目标变量
        # 线性层直接映射: L 个历史点 → pred_len 个未来点
        target_pred = self.target_linear(target).unsqueeze(-1)  # [B, pred_len, 1]

        # ==================== 分支二: 加噪特征抗噪轨迹 (Feature Track) ====================
        # ---- 步骤 1: 实例归一化 (Instance Normalization) ----
        # 目的: 消除不同特征间的量纲差异，防止幅值较大的特征主导模型学习
        # 注意: 这是手动实现的归一化（非 RevIN），均值/标准差在预测后需要反归一化
        means = x_enc.mean(1, keepdim=True)      # [B, 1, C]  - 每个特征沿时间维度的均值
        x = x_enc - means                         # 去均值 (中心化)
        stds = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        )                                         # [B, 1, C]  - 每个特征沿时间维度的标准差
        x /= stds                                 # 除以标准差 (标准化)

        # ---- 步骤 2: Patching & Embedding (CI 模式) ----
        # CI (Channel Independence) 策略:
        #   将 [B, L, C] 重塑为 [B*C, L, 1]，即每个变量被当作独立的一维序列
        #   这降低了建模复杂度，避免不同变量之间的复杂交互
        x = x.permute(0, 2, 1).reshape(B * C, L, 1)  # [B*C, L, 1]

        # Padding 处理:
        #   如果 L 不能被 seg_len 整除，在末尾用最后 pad_len 个值填充
        #   确保序列长度可以完整切分为整数个 patch
        if L % self.seg_len != 0:
            pad_len = self.seg_len - (L % self.seg_len)
            x = torch.cat([x, x[:, -pad_len:, :]], dim=1)  # [B*C, L+pad_len, 1]

        # 切片为 patches: 将长序列切分为 Seg_Num 个长度为 seg_len 的片段
        # 例如: L=192, seg_len=24 → Seg_Num = 8
        x = x.reshape(B * C, -1, self.seg_len)  # [B*C, Seg_Num, seg_len]

        # 每个 patch 独立嵌入: seg_len → d_model
        x = self.feature_embedding(x)  # [B*C, Seg_Num, d_model]

        # ---- 步骤 3: GRU 时序编码 ----
        # 对 patch 序列进行双向 GRU 编码，捕获跨 patch 的全局时序依赖
        # 输入:  [B*C, Seg_Num, d_model]   - patch 序列
        # h_n:   [2, B*C, d_model]         - 双向最后一层每方向的最终隐状态
        #        h_n[0] = 前向隐状态, h_n[1] = 反向隐状态
        # 注意: 我们只用最终隐状态 h_n，不使用输出序列（_），因为
        # 最终隐状态蕴含了整个序列的摘要信息，足够用于一步解码
        _, h_n = self.gru(x)  # h_n: [2, B*C, d_model]

        # 将双向隐状态拼接: [2, B*C, d_model] → [B*C, 2*d_model]
        h_n = h_n.permute(1, 0, 2).reshape(B * C, -1)  # [B*C, 2*d_model]

        # ---- 步骤 4: Direct Projection 解码 ----
        # 关键设计: 一次性将 GRU 隐状态映射到完整的 pred_len 预测
        # 不使用逐步自回归 (autoregressive) 解码，从而:
        #   - 避免误差在递归中逐步累积
        #   - 支持长达 720 步的预测窗口而不退化
        feat_pred = self.decode_proj(h_n)  # [B*C, pred_len]

        # 恢复 CI 前的维度: [B*C, pred_len] → [B, C, pred_len] → [B, pred_len, C]
        feat_pred = feat_pred.reshape(B, C, self.pred_len).permute(0, 2, 1)

        # ==================== 最终融合 (Fusion) ====================
        # ---- 反归一化 ----
        # 将 Feature Track 的输出恢复到原始数据尺度
        feat_pred = feat_pred * stds + means  # [B, pred_len, C]

        # ---- 双分支融合 (仅对目标变量，即最后一列) ----
        # combine_weight 为两个可学习标量，softmax 后转为概率分布
        # w[0]: Anchor Track 的权重，w[1]: Feature Track 的权重
        w = torch.softmax(self.combine_weight, dim=0)

        # 目标变量 = α * Anchor分支预测 + β * Feature分支预测
        # target_pred: [B, pred_len, 1]  (来自分支一)
        # feat_pred[:, :, -1:]: [B, pred_len, 1]  (来自分支二的最后一列)
        final_target = w[0] * target_pred + w[1] * feat_pred[:, :, -1:]

        # ---- 拼接输出 ----
        # 其他特征列 (除最后一列) 直接使用 Feature Track 的输出
        # 目标变量列使用双分支融合结果
        out = torch.cat([feat_pred[:, :, :-1], final_target], dim=-1)
        # 输出: [B, pred_len, C]

        return out
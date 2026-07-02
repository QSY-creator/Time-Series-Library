"""
生成 ECL 数据趋势-季节分解图 (第三章 SSSS 模型说明用)
利用 moving_avg(25) 进行序列分解，展示原始/趋势/季节三部分
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============ 1. 加载数据 ============
df = pd.read_csv('./dataset/electricity/electricity.csv')
# OT 是最后一列 (Dataset_Custom 会把 target 重排到末尾)
ot = df.iloc[:, -1].values.astype(np.float32)

# ============ 2. 选取一段 (前 600 小时，约 25 天) ============
n_show = 600
segment = ot[:n_show]

# ============ 3. 移动平均分解 ============
kernel_size = 25  # 与项目 scripts 中 moving_avg 参数一致
pad_size = kernel_size // 2

# 1D 移动平均 (padding 保持长度不变)
padded = np.pad(segment, (pad_size, pad_size), mode='edge')
trend = np.convolve(padded, np.ones(kernel_size) / kernel_size, mode='valid')
seasonal = segment - trend

# ============ 4. 绘图 ============
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

t = np.arange(n_show)

# 上图: 原始信号
axes[0].plot(t, segment, '#1F77B4', linewidth=1.2)
axes[0].set_ylabel('Load (kWh)')
axes[0].set_title('Original Time Series (ECL, OT Channel, First 600 Hours)', fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.3)

# 中图: 趋势分量
axes[1].plot(t, trend, '#D62728', linewidth=1.5)
axes[1].set_ylabel('Trend')
axes[1].set_title('Trend Component (moving_avg, kernel=25)', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

# 下图: 季节/波动分量
axes[2].plot(t, seasonal, '#2CA02C', linewidth=1.0)
axes[2].axhline(y=0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
axes[2].set_ylabel('Seasonal')
axes[2].set_xlabel('Time (hours)')
axes[2].set_title('Seasonal / Residual Component (Original - Trend)', fontsize=12, fontweight='bold')
axes[2].grid(alpha=0.3)

fig.tight_layout()
output_dir = './report_figures'
os.makedirs(output_dir, exist_ok=True)
fig.savefig(f'{output_dir}/fig_decomposition.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'Saved: {output_dir}/fig_decomposition.png')
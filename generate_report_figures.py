"""
生成实验报告所需的分析图表
从 exp_outputs 中提取指标数据并绘制对比图
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# 数据来源: exp_outputs/SSSS_ai_practice.pdf + txt 文件
# ============================================================

# ---------- Table 1: 单变量 (Univariate OT) ----------
pred_lens = [96, 192, 336, 720]

uv_data = {
    'SSSS (Ours)': {
        'MSE': [0.3034, 0.3353, 0.3638, 0.4231],
        'MAE': [0.3898, 0.4081, 0.4336, 0.4746],
        'SMAPE': [6.24, 6.51, 6.90, 7.57],
        'R2':   [0.7109, 0.6816, 0.6558, 0.5963],
    },
    'Pyraformer': {
        'MSE': [0.3070, 0.3275, 0.3687, 0.4563],
        'MAE': [0.3932, 0.4019, 0.4368, 0.4893],
        'SMAPE': [6.29, 6.41, 6.96, 7.79],
        'R2':   [0.7074, 0.6890, 0.6511, 0.5647],
    },
    'DLinear': {
        'MSE': [0.3873, 0.3645, 0.3908, 0.4277],
        'MAE': [0.4512, 0.4353, 0.4529, 0.4866],
        'SMAPE': [7.23, 6.95, 7.21, 7.75],
        'R2':   [0.6309, 0.6538, 0.6302, 0.5919],
    },
    'TSMixer': {
        'MSE': [0.3412, 0.3511, 0.3966, 0.4439],
        'MAE': [0.4210, 0.4267, 0.4562, 0.4946],
        'SMAPE': [6.77, 6.84, 7.30, 7.93],
        'R2':   [0.6749, 0.6666, 0.6247, 0.5764],
    },
    'XGBoost': {
        'MSE': [0.3491, 0.3367, 0.3672, 0.4060],
        'MAE': [0.4307, 0.4173, 0.4368, 0.4726],
        'SMAPE': [7.07, 6.84, 7.15, 7.69],
        'R2':   [0.6673, 0.6802, 0.6526, 0.6126],
    },
    'LinearRegression': {
        'MSE': [0.3712, 0.3506, 0.3780, 0.4156],
        'MAE': [0.4369, 0.4217, 0.4409, 0.4770],
        'SMAPE': [7.17, 6.91, 7.21, 7.76],
        'R2':   [0.6462, 0.6671, 0.6423, 0.6034],
    },
    'Ridge': {
        'MSE': [0.3712, 0.3506, 0.3780, 0.4156],
        'MAE': [0.4369, 0.4217, 0.4409, 0.4770],
        'SMAPE': [7.17, 6.91, 7.21, 7.76],
        'R2':   [0.6462, 0.6671, 0.6423, 0.6034],
    },
    'RandomForest': {
        'MSE': [0.5676, 0.5790, 0.5969, 0.6198],
        'MAE': [0.5796, 0.5900, 0.5996, 0.6081],
        'SMAPE': [9.51, 9.65, 9.77, 9.89],
        'R2':   [0.4591, 0.4501, 0.4352, 0.4086],
    },
}

# ---------- Table 2: 多变量 (Multivariate Non-OT) ----------
mv_data = {
    'SSSS (Ours)': {
        'MSE': [0.1654, 0.1751, 0.1920, 0.2313],
        'MAE': [0.2517, 0.2615, 0.2787, 0.3113],
        'SMAPE': [12.66, 12.91, 13.52, 14.72],
        'R2':   [0.8366, 0.8269, 0.8098, 0.7698],
    },
    'DLinear': {
        'MSE': [0.2104, 0.2102, 0.2231, 0.2578],
        'MAE': [0.3016, 0.3047, 0.3192, 0.3496],
        'SMAPE': [15.27, 15.30, 15.84, 17.09],
        'R2':   [0.7922, 0.7922, 0.7790, 0.7435],
    },
    'TSMixer': {
        'MSE': [0.2041, 0.2182, 0.2394, 0.2720],
        'MAE': [0.3082, 0.3289, 0.3502, 0.3730],
        'SMAPE': [15.85, 16.57, 17.39, 18.29],
        'R2':   [0.7983, 0.7843, 0.7629, 0.7294],
    },
    'Pyraformer': {
        'MSE': [0.2784, 0.2936, 0.2920, 0.2971],
        'MAE': [0.3737, 0.3893, 0.3883, 0.3857],
        'SMAPE': [18.78, 19.26, 19.13, 18.97],
        'R2':   [0.7250, 0.7097, 0.7107, 0.7043],
    },
}

# ============================================================
# 样式配置
# ============================================================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

output_dir = './report_figures'
os.makedirs(output_dir, exist_ok=True)

# 配色方案
colors_uv = {
    'SSSS (Ours)': '#D62728',
    'Pyraformer': '#1F77B4',
    'DLinear': '#2CA02C',
    'TSMixer': '#9467BD',
    'XGBoost': '#FF7F0E',
    'LinearRegression': '#8C564B',
    'Ridge': '#E377C2',
    'RandomForest': '#7F7F7F',
}
colors_mv = {
    'SSSS (Ours)': '#D62728',
    'DLinear': '#2CA02C',
    'TSMixer': '#9467BD',
    'Pyraformer': '#1F77B4',
}
markers_uv = {
    'SSSS (Ours)': 's',
    'Pyraformer': 'o',
    'DLinear': '^',
    'TSMixer': 'D',
    'XGBoost': 'v',
    'LinearRegression': '<',
    'Ridge': '>',
    'RandomForest': 'x',
}
markers_mv = {
    'SSSS (Ours)': 's',
    'DLinear': '^',
    'TSMixer': 'D',
    'Pyraformer': 'o',
}
line_styles = {
    'SSSS (Ours)': '-',
    'Pyraformer': '--',
    'DLinear': '-.',
    'TSMixer': ':',
    'XGBoost': '--',
    'LinearRegression': '-.',
    'Ridge': ':',
    'RandomForest': (0, (3, 2, 1, 2)),
}


def plot_metric_vs_predlen(data, metric_name, ylabel, title_prefix, colors, markers,
                           styles, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, metrics in data.items():
        color = colors.get(model_name, 'black')
        marker = markers.get(model_name, 'o')
        ls = styles.get(model_name, '-')
        ax.plot(pred_lens, metrics[metric_name],
                color=color, marker=marker, linestyle=ls,
                linewidth=2 if model_name == 'SSSS (Ours)' else 1.5,
                markersize=8 if model_name == 'SSSS (Ours)' else 6,
                label=model_name, zorder=5 if model_name == 'SSSS (Ours)' else 3,
                alpha=1.0 if model_name == 'SSSS (Ours)' else 0.85)
    ax.set_xlabel('Prediction Length (time steps)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title_prefix}: {metric_name} vs Prediction Length')
    ax.set_xticks(pred_lens)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/{filename}', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_dir}/{filename}')


def plot_dl_vs_ml_bar():
    """DL vs ML 分组柱状图 - 两组使用统一 Y 轴范围"""
    dl_models = ['SSSS (Ours)', 'Pyraformer', 'DLinear', 'TSMixer']
    ml_models = ['XGBoost', 'LinearRegression', 'Ridge', 'RandomForest']

    dl_avg_mse = [np.mean(uv_data[m]['MSE']) for m in dl_models]
    ml_avg_mse = [np.mean(uv_data[m]['MSE']) for m in ml_models]

    # 统一纵坐标范围：截断到 0.42 以放大非 RF 模型间的差异，RF 做截断突出
    y_min = 0.30
    y_cut = 0.42
    y_full = max(dl_avg_mse + ml_avg_mse) * 1.05

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_dl = np.arange(len(dl_models))
    x_ml = np.arange(len(ml_models))
    width = 0.6

    # DL 子图
    ax = axes[0]
    bars1 = ax.bar(x_dl, dl_avg_mse, width,
                   color=['#D62728', '#1F77B4', '#2CA02C', '#9467BD'],
                   edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_dl)
    ax.set_xticklabels(dl_models, rotation=20, ha='right')
    ax.set_ylabel('Average MSE')
    ax.set_title('Deep Learning Models')
    ax.set_ylim(y_min, y_cut)
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars1, dl_avg_mse)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        if i == 0:  # SSSS — 最佳标记
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.010,
                    '★ Best', ha='center', va='bottom', fontsize=12,
                    fontweight='bold', color='#DAA520')

    # ML 子图
    ax = axes[1]
    # RandomForest 柱体特殊处理：截断+斜线+红色边框
    bars2 = ax.bar(x_ml, ml_avg_mse, width,
                   color=['#FF7F0E', '#8C564B', '#E377C2', '#FF4444'],
                   edgecolor=['black', 'black', 'black', '#CC0000'],
                   linewidth=[0.5, 0.5, 0.5, 2.0])
    # RandomForest 柱体添加斜线填充
    bars2[3].set_hatch('////')
    ax.set_xticks(x_ml)
    ax.set_xticklabels(ml_models, rotation=20, ha='right')
    ax.set_ylabel('Average MSE')
    ax.set_title('Traditional ML Models')
    ax.set_ylim(y_min, y_cut)
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars2, ml_avg_mse)):
        if i == 3:  # RandomForest
            # 截断标注
            ax.text(bar.get_x() + bar.get_width()/2., y_cut - 0.004,
                    f'{val:.4f}', ha='center', va='top', fontsize=10,
                    fontweight='bold', color='#CC0000',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#CC0000', alpha=0.9))
            # 顶部锯齿标记
            ax.annotate('', xy=(bar.get_x() + bar.get_width()/2., y_cut + 0.001),
                        xytext=(bar.get_x() + bar.get_width()/2., y_cut - 0.015),
                        arrowprops=dict(arrowstyle='-', linestyle='--', color='#CC0000'))
            ax.text(bar.get_x() + bar.get_width()/2., y_cut + 0.003,
                    f'MSE={val:.4f}', ha='center', va='bottom', fontsize=8, color='#CC0000',
                    fontstyle='italic')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Univariate Forecasting: Average MSE Comparison (DL vs ML)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{output_dir}/fig_dl_vs_ml_bar.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_dir}/fig_dl_vs_ml_bar.png')


def plot_mv_comparison():
    """多变量: MSE/R² 双面板对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for metric_name, ax, ylabel in [('MSE', axes[0], 'MSE'), ('R2', axes[1], 'R²')]:
        for model_name, metrics in mv_data.items():
            color = colors_mv.get(model_name, 'black')
            marker = markers_mv.get(model_name, 'o')
            ax.plot(pred_lens, metrics[metric_name],
                    color=color, marker=marker, linestyle='-',
                    linewidth=2.5 if model_name == 'SSSS (Ours)' else 1.5,
                    markersize=8 if model_name == 'SSSS (Ours)' else 6,
                    label=model_name)
        ax.set_xlabel('Prediction Length')
        ax.set_ylabel(ylabel)
        ax.set_xticks(pred_lens)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    fig.suptitle('Multivariate (Non-OT) Forecasting: MSE & R² Comparison',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{output_dir}/fig_mv_mse_r2.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_dir}/fig_mv_mse_r2.png')


def plot_mse_growth():
    """各模型 MSE 衰减率对比 (96 -> 720)"""
    def calc_growth(data_dict):
        growth = {}
        for name, d in data_dict.items():
            growth[name] = (d['MSE'][-1] - d['MSE'][0]) / d['MSE'][0] * 100
        return growth

    uv_growth = calc_growth(uv_data)
    mv_growth = calc_growth(mv_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Univariate
    names_uv = list(uv_growth.keys())
    vals_uv = [uv_growth[n] for n in names_uv]
    colors_uv_list = [colors_uv.get(n, 'gray') for n in names_uv]
    axes[0].barh(names_uv, vals_uv, color=colors_uv_list, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('MSE Increase from 96 to 720 (%)')
    axes[0].set_title('Univariate (OT)')
    axes[0].grid(axis='x', alpha=0.3)
    for i, v in enumerate(vals_uv):
        axes[0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

    # Multivariate
    names_mv = list(mv_growth.keys())
    vals_mv = [mv_growth[n] for n in names_mv]
    colors_mv_list = [colors_mv.get(n, 'gray') for n in names_mv]
    axes[1].barh(names_mv, vals_mv, color=colors_mv_list, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('MSE Increase from 96 to 720 (%)')
    axes[1].set_title('Multivariate (Non-OT)')
    axes[1].grid(axis='x', alpha=0.3)
    for i, v in enumerate(vals_mv):
        axes[1].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

    fig.suptitle('MSE Growth Rate (96→720): Lower is Better', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{output_dir}/fig_mse_growth.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_dir}/fig_mse_growth.png')


if __name__ == '__main__':
    print('Generating report figures...')
    plot_metric_vs_predlen(uv_data, 'MSE', 'MSE', 'Univariate (OT)',
                           colors_uv, markers_uv, line_styles, 'fig_uv_mse.png')
    plot_metric_vs_predlen(uv_data, 'R2', 'R²', 'Univariate (OT)',
                           colors_uv, markers_uv, line_styles, 'fig_uv_r2.png')
    plot_metric_vs_predlen(uv_data, 'SMAPE', 'SMAPE (%)', 'Univariate (OT)',
                           colors_uv, markers_uv, line_styles, 'fig_uv_smape.png')
    plot_metric_vs_predlen(uv_data, 'MAE', 'MAE', 'Univariate (OT)',
                           colors_uv, markers_uv, line_styles, 'fig_uv_mae.png')
    plot_dl_vs_ml_bar()
    plot_mv_comparison()
    plot_mse_growth()
    print(f'\nAll figures saved to {output_dir}/')
    for f in sorted(os.listdir(output_dir)):
        print(f'  {f}')
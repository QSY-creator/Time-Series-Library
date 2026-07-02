"""
生成真实值 vs 预测值对比曲线 (图 5.7)
从已保存的 prediction results 中加载 pred.npy & true.npy
优先使用 SSSS 结果，回退到 TSMixer
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob

def find_result_dirs(base_dir, model_name):
    """查找模型的所有 pred_len 结果目录"""
    pattern = os.path.join(base_dir, 'results', '*')
    dirs = sorted(glob.glob(pattern))
    result_dirs = {}
    for d in dirs:
        dname = os.path.basename(d)
        for pl in [96, 192, 336, 720]:
            if f'pl{pl}' in dname:
                result_dirs[pl] = d
    return result_dirs


def load_pred_true(result_dir, ot_only=True):
    """加载 pred.npy 和 true.npy，可选仅提取 OT 通道
    返回 (pred, true) 或 (None, None) 表示加载失败
    """
    pred_path = os.path.join(result_dir, 'pred.npy')
    true_path = os.path.join(result_dir, 'true.npy')
    try:
        pred = np.load(pred_path, allow_pickle=True)
        true = np.load(true_path, allow_pickle=True)
    except (ValueError, OSError) as e:
        print(f'  ⚠️  Skipped (corrupted file): {os.path.basename(pred_path)}: {e}')
        return None, None
    if ot_only and pred.ndim >= 2:
        pred = pred[:, :, -1]  # OT 是最后一列
        true = true[:, :, -1]
    return pred, true


def plot_comparison(pred, true, pred_len, model_name, save_path, n_steps=200):
    """
    绘制真实值 vs 预测值对比曲线
    pred/true: [N, pred_len]  或 [N_total]
    """
    # 如果是多步预测，展平并选取最后 n_steps 个点
    if pred.ndim == 2:
        # 取最后一个样本 + 前几个样本的拼接
        n_samples = pred.shape[0]
        # 展平为连续序列
        pred_flat = pred[-min(n_samples, 10):].reshape(-1)
        true_flat = true[-min(n_samples, 10):].reshape(-1)
    else:
        pred_flat = pred
        true_flat = true

    # 截取最后 n_steps 个点
    total = min(len(pred_flat), n_steps)
    pred_show = pred_flat[-total:]
    true_show = true_flat[-total:]

    fig, ax = plt.subplots(figsize=(14, 5))

    t = np.arange(total)
    ax.plot(t, true_show, 'b-', linewidth=1.5, label='Ground Truth', alpha=0.8)
    ax.plot(t, pred_show, 'r--', linewidth=1.5, label=f'{model_name} Prediction', alpha=0.85)

    # 计算该段的指标
    mse = np.mean((pred_show - true_show) ** 2)
    mae = np.mean(np.abs(pred_show - true_show))
    # SMAPE
    denom = (np.abs(pred_show) + np.abs(true_show)) / 2.0
    smape = np.mean(np.abs(pred_show - true_show) / denom) * 100

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Normalized Load', fontsize=12)
    ax.set_title(f'{model_name} - Prediction Length = {pred_len}\n'
                 f'MSE={mse:.4f}  MAE={mae:.4f}  SMAPE={smape:.2f}%',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved: {save_path}')
    return {'MSE': mse, 'MAE': mae, 'SMAPE': smape}


if __name__ == '__main__':
    output_dir = './report_figures'
    os.makedirs(output_dir, exist_ok=True)

    # =============================================
    # 尝试多个模型源，优先 SSSS
    # =============================================
    model_sources = [
        ('SSSS', './exp_outputs/ECL_SSSS_S_OT'),
        ('SSSS', './exp_outputs/ECL_SSSS'),
        ('TSMixer', './exp_outputs/ECL_TSMixer'),
    ]

    for model_name, base_dir in model_sources:
        result_dirs = find_result_dirs(base_dir, model_name)
        if len(result_dirs) >= 2:
            print(f'\n>>> Using {model_name} results from {base_dir}')
            print(f'    Available pred_lens: {sorted(result_dirs.keys())}')

            # 选 96 和 720 两种极端场景
            # Select available pred_lens: prefer 96 and 720, fallback to whatever works
            target_lens = [96, 720, 336, 192]
            plotted = 0
            for pred_len in target_lens:
                if pred_len in result_dirs and plotted < 2:
                    d = result_dirs[pred_len]
                    print(f'  Loading pred_len={pred_len} from {d}')
                    pred, true = load_pred_true(d, ot_only=True)
                    if pred is None:
                        continue
                    save_path = f'{output_dir}/fig_pred_curve_{model_name}_pl{pred_len}.png'
                    metrics = plot_comparison(pred, true, pred_len, model_name, save_path, n_steps=250)
                    plotted += 1
            break
    else:
        print('\n⚠️  No model results found with pred.npy/true.npy files.')
        print('   To generate SSSS predictions, run:')
        print('     bash scripts/long_term_forecast/ECL_script/SSSS_S_OT.sh')
        print('   Or use the TSMixer eval script:')
        print('     bash scripts/long_term_forecast/ECL_script/TSMixer_M_eval.sh')

    print(f'\nDone. Figures saved to {output_dir}/')
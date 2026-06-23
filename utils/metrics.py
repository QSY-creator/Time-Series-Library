import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def _mask_near_zero(true, ratio=0.01):
    """生成 mask：保留 |true| 大于列均值 ratio 倍的样本。

    对多变量序列（最后一维为变量/channel），每列独立计算阈值，
    避免尺度差异大的列互相影响。主要用于 electricity 这类含 0 的数据集。
    """
    abs_true = np.abs(true)
    # 对除最后一维外的所有维度求均值，得到每列的 mean(|true|)
    reduce_axes = tuple(range(abs_true.ndim - 1))
    col_mean = np.mean(abs_true, axis=reduce_axes, keepdims=True)
    threshold = ratio * col_mean
    return abs_true > threshold


def MAPE(pred, true, zero_mask_ratio=0.01):
    mask = _mask_near_zero(true, ratio=zero_mask_ratio)
    if not np.any(mask):
        return float('nan')
    abs_true = np.abs(true[mask])
    return np.mean(np.abs((true[mask] - pred[mask]) / abs_true))


def MSPE(pred, true, zero_mask_ratio=0.01):
    mask = _mask_near_zero(true, ratio=zero_mask_ratio)
    if not np.any(mask):
        return float('nan')
    abs_true = np.abs(true[mask])
    return np.mean(np.square((true[mask] - pred[mask]) / abs_true))


def SMAPE(pred, true):
    """Symmetric Mean Absolute Percentage Error.

    对 true=0 不敏感，适合 electricity 这类含大量近零值的多变量序列。
    """
    denominator = (np.abs(pred) + np.abs(true)) / 2.0
    # denominator 为 0 只有当 pred 和 true 同时为 0，此时误差定义为 0
    mask = denominator > 0
    if not np.any(mask):
        return float('nan')
    return np.mean(np.abs(pred[mask] - true[mask]) / denominator[mask])


def R2(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    epsilon = 1e-10
    if ss_tot < epsilon:
        return float('nan')
    return 1 - ss_res / ss_tot


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = R2(pred, true)

    return mae, mse, rmse, mape, mspe, r2

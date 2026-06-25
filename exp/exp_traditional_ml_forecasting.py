from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import visual
from utils.metrics import metric, SMAPE
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pickle
from utils.dtw_metric import accelerated_dtw

warnings.filterwarnings('ignore')


def _get_sklearn_model(ml_model_name, args):
    """根据名称构造 sklearn 回归器。"""
    if ml_model_name == 'LinearRegression':
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    elif ml_model_name == 'Ridge':
        from sklearn.linear_model import Ridge
        return Ridge(alpha=args.ml_alpha if hasattr(args, 'ml_alpha') else 1.0)
    elif ml_model_name == 'Lasso':
        from sklearn.linear_model import Lasso
        return Lasso(alpha=args.ml_alpha if hasattr(args, 'ml_alpha') else 0.1)
    elif ml_model_name == 'RandomForest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=args.ml_n_estimators if hasattr(args, 'ml_n_estimators') else 10,
            max_depth=args.ml_max_depth if hasattr(args, 'ml_max_depth') else 5,
            min_samples_split=20,
            min_samples_leaf=10,
            max_samples=0.5,
            n_jobs=1,
            random_state=2021
        )
    elif ml_model_name == 'GradientBoosting':
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=args.ml_n_estimators if hasattr(args, 'ml_n_estimators') else 100,
            random_state=2021
        )
    elif ml_model_name == 'XGBoost':
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=args.ml_n_estimators if hasattr(args, 'ml_n_estimators') else 10,
            max_depth=args.ml_max_depth if hasattr(args, 'ml_max_depth') else 3,
            subsample=args.ml_subsample if hasattr(args, 'ml_subsample') else 0.5,
            colsample_bytree=args.ml_colsample_bytree if hasattr(args, 'ml_colsample_bytree') else 0.5,
            n_jobs=1,
            random_state=2021
        )
    else:
        raise ValueError(f'Unsupported ml_model: {ml_model_name}')


def _wrap_multioutput(model, ml_model_name):
    """对非原生多输出模型使用 MultiOutputRegressor 包装。"""
    # 以下模型原生支持多输出回归，无需包装
    if ml_model_name in ['LinearRegression', 'Ridge', 'Lasso', 'RandomForest', 'XGBoost']:
        return model
    else:
        from sklearn.multioutput import MultiOutputRegressor
        return MultiOutputRegressor(model, n_jobs=-1)


class Exp_Traditional_ML_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Traditional_ML_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # 传统 ML 模型不需要 PyTorch 优化器，但 Exp_Basic 初始化要求至少一个可训练参数
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _target_col_index(self, data_set):
        """确定目标变量在 data_x 中的列索引。

        Dataset_Custom 已经把 target 重排到最后一列，因此默认返回 -1。
        若数据加载器保留了原始列顺序，可通过 args.target 查找。
        """
        target = getattr(self.args, 'target', 'OT')
        # 优先尝试从 data_set 的列名中定位
        columns = getattr(data_set, 'columns', None)
        if columns is not None:
            cols = list(columns)
            if target in cols:
                return cols.index(target)
        # 回退：Dataset_Custom 保证 target 在最后一列
        return -1

    def _make_window_indices(self, total_len, window_len):
        """生成滑动窗口索引，复用以避免逐通道重复构造。"""
        start_indices = np.arange(total_len - window_len + 1)
        return start_indices[:, None] + np.arange(window_len)[None, :]

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        ml_model_name = getattr(self.args, 'ml_model', 'Ridge')
        print(f'\n>>> Training traditional ML model: {ml_model_name}')

        data_full = train_data.data_x.astype(np.float32)
        total_len = len(data_full)
        window_len = self.args.seq_len + self.args.pred_len
        window_indices = self._make_window_indices(total_len, window_len)
        num_channels = data_full.shape[-1]
        num_samples = total_len - window_len + 1
        print(f'Training samples: {num_samples}, channels: {num_channels}, window_len: {window_len}')

        if self.args.features == 'MS':
            # 多变量输入 -> 单目标输出
            target_col = self._target_col_index(train_data)
            print(f'MS mode: use all {num_channels} channels as input, predict target col {target_col}')

            windows = data_full[window_indices]  # [N, window_len, C]
            X = windows[:, :self.args.seq_len, :].reshape(num_samples, -1)  # [N, seq_len*C]
            Y = windows[:, self.args.seq_len:, target_col]  # [N, pred_len]

            base_model = _get_sklearn_model(ml_model_name, self.args)
            model_c = _wrap_multioutput(base_model, ml_model_name)
            fit_start = time.time()
            model_c.fit(X, Y)
            print(f'  Fitted single model, cost {time.time()-fit_start:.2f}s')

            with open(os.path.join(path, 'model_0.pkl'), 'wb') as f:
                pickle.dump(model_c, f)
            del model_c
        else:
            # M 模式：逐通道独立预测
            for c in range(num_channels):
                channel_start = time.time()
                channel_data = data_full[:, c]
                windows = channel_data[window_indices]  # [N, window_len]
                X = windows[:, :self.args.seq_len]  # [N, seq_len]
                Y = windows[:, self.args.seq_len:]  # [N, pred_len]

                base_model = _get_sklearn_model(ml_model_name, self.args)
                model_c = _wrap_multioutput(base_model, ml_model_name)
                model_c.fit(X, Y)

                # 逐通道保存，避免大模型（RandomForest/XGBoost）同时驻留内存
                with open(os.path.join(path, f'model_{c}.pkl'), 'wb') as f:
                    pickle.dump(model_c, f)
                del model_c

                if (c + 1) % 10 == 0 or c == num_channels - 1:
                    print(f'  Fitted channel {c+1}/{num_channels}, cost {time.time()-channel_start:.2f}s')

        print(f'Models saved to {path}/model_*.pkl\n')
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        ml_model_name = getattr(self.args, 'ml_model', 'Ridge')

        # result save
        folder_path = os.path.join(self.args.output_base, 'results', setting) + '/'
        os.makedirs(folder_path, exist_ok=True)

        print('Testing on test set...')
        num_channels = test_data.data_x.shape[-1]
        n_total = 0
        sum_abs_err = 0.0
        sum_sq_err = 0.0
        sum_true = 0.0
        sum_true_sq = 0.0
        sum_smape_num = 0.0
        sum_smape_den = 0.0

        if self.args.features == 'MS':
            # 多变量输入 -> 单目标输出
            target_col = self._target_col_index(test_data)
            print(f'MS mode: predict target col {target_col}')

            with open(os.path.join(path, 'model_0.pkl'), 'rb') as f:
                model_c = pickle.load(f)

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().cpu().numpy()
                batch_y = batch_y.float().cpu().numpy()

                X = batch_x.reshape(batch_x.shape[0], -1).astype(np.float32)  # [B, seq_len*C]
                pred = model_c.predict(X).astype(np.float32)  # [B, pred_len]
                true = batch_y[:, -self.args.pred_len:, target_col].astype(np.float32)  # [B, pred_len]

                if test_data.scale and not self.args.inverse:
                    # 单通道反标准化
                    scale_c = test_data.scaler.scale_[target_col]
                    mean_c = test_data.scaler.mean_[target_col]
                    pred_raw = pred * scale_c + mean_c
                    true_raw = true * scale_c + mean_c
                else:
                    pred_raw, true_raw = pred, true

                n_total += pred.size
                diff = pred - true
                sum_abs_err += np.abs(diff).sum()
                sum_sq_err += (diff ** 2).sum()
                sum_true += true.sum()
                sum_true_sq += (true ** 2).sum()

                smape_den = (np.abs(pred_raw) + np.abs(true_raw)) / 2.0
                smape_mask = smape_den > 0
                sum_smape_num += np.abs(pred_raw[smape_mask] - true_raw[smape_mask]).sum()
                sum_smape_den += smape_den[smape_mask].sum()

            del model_c
        else:
            # M 模式：逐通道独立预测
            # 轻量模型一次性加载；树模型按通道加载以控制内存
            lightweight_models = ['LinearRegression', 'Ridge', 'Lasso']
            if ml_model_name in lightweight_models:
                models = []
                for c in range(num_channels):
                    with open(os.path.join(path, f'model_{c}.pkl'), 'rb') as f:
                        models.append(pickle.load(f))

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().cpu().numpy()
                    batch_y = batch_y.float().cpu().numpy()
                    true_full = batch_y[:, -self.args.pred_len:, :].astype(np.float32)
                    batch_size = batch_x.shape[0]

                    pred_full = np.zeros((batch_size, self.args.pred_len, num_channels), dtype=np.float32)
                    for c in range(num_channels):
                        X = batch_x[:, :, c].astype(np.float32)
                        pred_full[:, :, c] = models[c].predict(X)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    pred = pred_full[:, :, f_dim:]
                    true = true_full[:, :, f_dim:]

                    if test_data.scale and not self.args.inverse:
                        shape = pred.shape
                        pred_raw = test_data.inverse_transform(pred.reshape(-1, pred.shape[-1])).reshape(shape)
                        true_raw = test_data.inverse_transform(true.reshape(-1, true.shape[-1])).reshape(shape)
                    else:
                        pred_raw, true_raw = pred, true

                    n_total += pred.size
                    diff = pred - true
                    sum_abs_err += np.abs(diff).sum()
                    sum_sq_err += (diff ** 2).sum()
                    sum_true += true.sum()
                    sum_true_sq += (true ** 2).sum()

                    smape_den = (np.abs(pred_raw) + np.abs(true_raw)) / 2.0
                    smape_mask = smape_den > 0
                    sum_smape_num += np.abs(pred_raw[smape_mask] - true_raw[smape_mask]).sum()
                    sum_smape_den += smape_den[smape_mask].sum()
            else:
                # 树模型：逐通道加载、预测、释放
                for c in range(num_channels):
                    channel_start = time.time()
                    with open(os.path.join(path, f'model_{c}.pkl'), 'rb') as f:
                        model_c = pickle.load(f)

                    _, test_loader_c = self._get_data(flag='test')
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader_c):
                        batch_x = batch_x.float().cpu().numpy()
                        batch_y = batch_y.float().cpu().numpy()

                        X = batch_x[:, :, c].astype(np.float32)
                        pred = model_c.predict(X).astype(np.float32)
                        true = batch_y[:, -self.args.pred_len:, c].astype(np.float32)

                        if test_data.scale and not self.args.inverse:
                            scale_c = test_data.scaler.scale_[c]
                            mean_c = test_data.scaler.mean_[c]
                            pred_raw = pred * scale_c + mean_c
                            true_raw = true * scale_c + mean_c
                        else:
                            pred_raw, true_raw = pred, true

                        n_total += pred.size
                        diff = pred - true
                        sum_abs_err += np.abs(diff).sum()
                        sum_sq_err += (diff ** 2).sum()
                        sum_true += true.sum()
                        sum_true_sq += (true ** 2).sum()

                        smape_den = (np.abs(pred_raw) + np.abs(true_raw)) / 2.0
                        smape_mask = smape_den > 0
                        sum_smape_num += np.abs(pred_raw[smape_mask] - true_raw[smape_mask]).sum()
                        sum_smape_den += smape_den[smape_mask].sum()

                    del model_c
                    if (c + 1) % 10 == 0 or c == num_channels - 1:
                        print(f'  Tested channel {c+1}/{num_channels}, cost {time.time()-channel_start:.2f}s')

        mae = sum_abs_err / n_total
        mse = sum_sq_err / n_total
        rmse = np.sqrt(mse)
        ss_tot = sum_true_sq - (sum_true ** 2) / n_total
        r2 = 1 - sum_sq_err / ss_tot if ss_tot > 1e-10 else float('nan')
        smape = sum_smape_num / sum_smape_den if sum_smape_den > 0 else float('nan')

        dtw = 'Not calculated'

        print('mse:{:.6f}, mae:{:.6f}, smape:{:.4f}%, r2:{:.6f}, dtw:{}'.format(
            mse, mae, smape * 100, r2, dtw))
        file_name = os.path.join(self.args.output_base, 'result_long_term_forecast_{}.txt'.format(self.args.des))
        f = open(file_name, 'a')
        f.write(setting + "  \n")
        f.write('mse:{:.6f}, mae:{:.6f}, rmse:{:.6f}, smape:{:.4f}%, r2:{:.6f}, dtw:{}'.format(
            mse, mae, rmse, smape * 100, r2, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, smape, 0.0, r2]))

        return

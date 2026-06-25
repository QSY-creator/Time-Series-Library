import torch
from torch import nn


class Model(nn.Module):
    """
    传统机器学习模型（scikit-learn）的统一接口占位类。

    真实的训练与推理逻辑放在 exp_traditional_ml_forecasting.py 中，
    这里仅作为 Exp_Basic.model_dict 注册与 _select_optimizer 的接口入口。
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # 用于满足 PyTorch 优化器至少需要一个可训练参数的要求
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # traditional_ml_forecast 任务不通过此处进行真实推理
        return self.dummy

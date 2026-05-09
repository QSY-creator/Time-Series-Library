import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, KANAD, MSGNet, TimeFilter, Sundial, TimeMoE, Chronos, Moirai, TiRex,\
    TimesFM, Chronos2, RobustTimeMixer, RTimeMixer2, RTimeMixer3,DUETTimeMixer,DP_RTM,Super_DP,\
    Super_DP_RTM,Seg_DP_RTM,S_DP_RTM,Super_DR,SDP_Net,SDP_Net2,W_SDR,SSS,SSSS,S5,SSSS_wogru,SSSS_wolinear,SSSS_wopatching,SSSS_worevin


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer,
            'KANAD': KANAD,
            'MSGNet': MSGNet,
            'TimeFilter': TimeFilter,
            'Sundial': Sundial,
            'TimeMoE': TimeMoE,
            'Chronos': Chronos,
            'Moirai': Moirai,
            'TiRex': TiRex,
            'TimesFM': TimesFM,
            'Chronos2': Chronos2,
            'RobustTimeMixer': RobustTimeMixer,
            'RTimeMixer2': RTimeMixer2,
            'RTimeMixer3': RTimeMixer3,
            'DUETTimeMixer': DUETTimeMixer,
            'DP_RTM': DP_RTM,
            'Super_DP': Super_DP,
            'Super_DP_RTM': Super_DP_RTM,
            'Seg_DP_RTM': Seg_DP_RTM,
            'S_DP_RTM': S_DP_RTM,
            'Super_DR': Super_DR,
            'SDP_Net': SDP_Net,
            'SDP_Net2': SDP_Net2,
            'W_SDR': W_SDR,
            'SSS': SSS,
            'SSSS': SSSS,
            'S5': S5,
            'SSSS_wogru': SSSS_wogru,
            'SSSS_wolinear': SSSS_wolinear, 
            'SSSS_wopatching': SSSS_wopatching,
            'SSSS_worevin': SSSS_worevin
            
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self._print_model_params()

    def _build_model(self):
        raise NotImplementedError
        return None

    def _print_model_params(self):
        """计算并打印模型参数量"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        print('=' * 80)
        print(f'Model: {self.args.model}')
        print(f'Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)')
        print(f'Trainable Parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)')
        print(f'Non-trainable Parameters: {non_trainable_params:,} ({non_trainable_params / 1e6:.2f}M)')
        print('=' * 80)

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

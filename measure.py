import torch
import time
from thop import profile # 需要 pip install thop

def measure_efficiency(model, configs, device='cuda'):
    model.eval()
    model.to(device)
    
    # 创建伪数据 (Batch, Seq_Len, Channels)
    # 假设 Input=96, Channels=7 (ETTh1)
    dummy_input = torch.randn(1, configs.seq_len, configs.enc_in).to(device)
    # 对于 DeRNN，你还需要 x_mark_enc 等，这里简化为全0
    dummy_mark = torch.zeros(1, configs.seq_len, 4).to(device) 
    
    # 1. 计算 FLOPs 和 Params
    # 注意：需要根据模型 forward 的参数调整 inputs
    macs, params = profile(model, inputs=(dummy_input, dummy_mark, None, None))
    print(f"FLOPs: {macs / 1e9:.3f} G")
    print(f"Params: {params / 1e6:.3f} M")
    
    # 2. 计算 Inference Time (务必使用 torch.cuda.synchronize)
    # 预热 GPU
    for _ in range(100):
        _ = model(dummy_input, dummy_mark, None, None)
        
    start_time = time.time()
    torch.cuda.synchronize()
    iterations = 1000
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input, dummy_mark, None, None)
    torch.cuda.synchronize()
    end_time = time.time()
    
    latency = (end_time - start_time) / iterations * 1000 # 毫秒
    print(f"Inference Time: {latency:.3f} ms")
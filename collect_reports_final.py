import pandas as pd
import os

RESULT_FILE = 'result_long_term_forecast.txt' # TSLib 默认生成的汇总文件
if not os.path.exists(RESULT_FILE):
    print(f"找不到 {RESULT_FILE}，请确认实验是否运行成功")
    exit()

data = []
with open(RESULT_FILE, 'r') as f:
    lines = f.readlines()
    for line in lines:
        # 格式通常是: model_id, mse, mae ...
        # 或者: setting: ... \n mse:0.123, mae:0.234
        # 我们假设你用的是标准 TSLib，一行一条记录
        parts = line.strip().split(',')
        if len(parts) < 3: continue
        
        # 简单解析
        setting_str = parts[0]
        mse = float(parts[1].split(':')[-1]) if ':' in parts[1] else float(parts[1])
        mae = float(parts[2].split(':')[-1]) if ':' in parts[2] else float(parts[2])
        
        # 从 setting_str 里反解出模型和噪声
        # model_id 格式: ETTh1_Gaussian_Autoformer_MS
        
        noise_cat = 'Clean'
        if '_Gaussian_' in setting_str: noise_cat = 'Gaussian'
        elif '_Drift_' in setting_str: noise_cat = 'Drift'
        elif '_Dropout_' in setting_str: noise_cat = 'Dropout'
        
        # 提取模型名称
        model_name = "Unknown"
        # 这里需要更智能的提取，或者依赖之前定义的 model_id
        for m in ['Autoformer', 'Crossformer', 'iTransformer', 'Pyraformer', 'MICN', 'LightTS', 'TSMixer']:
            if f"_{m}_" in setting_str:
                model_name = m
                break
        
        # 提取数据集
        dataset_name = "Unknown"
        for d in ['ETTh1', 'ETTh2', 'ETTm1', 'Traffic', 'Electricity']:
            if d in setting_str:
                dataset_name = d
                break
                
        data.append({
            'Model': model_name,
            'Dataset': dataset_name,
            'Noise': noise_cat,
            'MSE': mse,
            'MAE': mae
        })

df = pd.DataFrame(data)

# 分割并保存
cats = ['Clean', 'Gaussian', 'Drift', 'Dropout']
for cat in cats:
    sub_df = df[df['Noise'] == cat].sort_values(by=['Dataset', 'Model'])
    sub_df.to_csv(f'Report_{cat}.csv', index=False)
    print(f"已生成 Report_{cat}.csv")
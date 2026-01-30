import pandas as pd
import os
import re

# 结果文件路径
RESULT_FILE = 'result_long_term_forecast.txt'
OUTPUT_DIR = './final_reports/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def parse_results():
    if not os.path.exists(RESULT_FILE):
        print(f"❌ 找不到 {RESULT_FILE}")
        return []

    parsed_data = []
    
    with open(RESULT_FILE, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # 解析格式: model_id, mse, mae
        # 示例: ETTh1_Gaussian_Autoformer_pl96_MS, 0.345, 0.456
        parts = line.split(',')
        if len(parts) < 3: continue
        
        setting = parts[0]
        try:
            mse = float(parts[1].split(':')[-1]) if ':' in parts[1] else float(parts[1])
            mae = float(parts[2].split(':')[-1]) if ':' in parts[2] else float(parts[2])
        except:
            continue

        # == 反向解析元数据 ==
        # 1. 噪声类型
        noise = 'Clean'
        if '_Gaussian_' in setting: noise = 'Gaussian'
        elif '_Drift_' in setting: noise = 'Drift'
        elif '_Dropout_' in setting: noise = 'Dropout'
        # 如果不是上面三种，且不含Clean字样，默认为Clean
        
        # 2. 数据集
        dataset = 'Unknown'
        for d in ['ETTh1', 'ETTh2', 'ETTm1', 'Traffic', 'Electricity']:
            if d in setting:
                dataset = d
                break
        
        # 3. 预测长度 (pl96)
        pred_len = 96
        pl_match = re.search(r'_pl(\d+)_', setting)
        if pl_match:
            pred_len = int(pl_match.group(1))
            
        # 4. 模型名称
        model = 'Unknown'
        # 必须按顺序匹配，防止 Pyraformer 匹配到 Former
        candidates = ['Autoformer', 'Crossformer', 'TimeMixer', 'iTransformer', 
                      'Pyraformer', 'LightTS', 'MICN', 'SSSS']
        for m in candidates:
            if f"_{m}_" in setting:
                model = m
                break
        
        parsed_data.append({
            'Dataset': dataset,
            'Model': model,
            'Noise': noise,
            'PredLen': pred_len,
            'MSE': mse,
            'MAE': mae
        })
        
    return parsed_data

data = parse_results()
if not data:
    print("没有解析到数据，请检查实验是否运行成功。")
else:
    df = pd.DataFrame(data)
    
    # 拆分为 4 个报告
    cats = ['Clean', 'Gaussian', 'Drift', 'Dropout']
    
    for cat in cats:
        # 筛选对应噪声
        sub_df = df[df['Noise'] == cat]
        
        if sub_df.empty:
            print(f"⚠️ {cat} 没有数据")
            continue
            
        # 数据透视表
        # 行: Dataset, Model
        # 列: PredLen (96, 192, 336, 720)
        # 值: MSE (你可以改成 MAE 或者同时显示)
        pivot = sub_df.pivot_table(
            index=['Dataset', 'Model'], 
            columns='PredLen', 
            values='MSE'
        )
        
        # 保存
        file_path = os.path.join(OUTPUT_DIR, f'Performance_{cat}_MSE.csv')
        pivot.to_csv(file_path)
        print(f"✅ 生成报表: {file_path}")

print("全部完成。")
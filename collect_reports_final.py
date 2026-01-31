import pandas as pd
import os

RESULT_FILE = 'result_long_term_forecast.txt'
if not os.path.exists(RESULT_FILE):
    print(f"找不到 {RESULT_FILE}，请确认实验是否运行成功")
    exit()

data = []
current_setting = None  # 用于暂存上一行的配置信息

with open(RESULT_FILE, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line: continue

        # --- 判断当前行是“结果行”还是“配置行” ---
        
        # 如果包含 mse: 和 mae:，说明这是结果行
        if 'mse:' in line and 'mae:' in line:
            if current_setting is None:
                # 如果没有对应的配置行，说明数据不完整，跳过
                continue
            
            # 1. 解析结果数值
            # line 格式: mse:0.249..., mae:0.385..., dtw:Not calculated
            parts = [p.strip() for p in line.split(',')]
            mse = None
            mae = None
            
            try:
                for p in parts:
                    if p.startswith('mse:'):
                        mse_val = p.split(':')[-1]
                        # 检查是否为 Not calculated
                        if 'Not calculated' not in mse_val:
                            mse = float(mse_val)
                    elif p.startswith('mae:'):
                        mae_val = p.split(':')[-1]
                        if 'Not calculated' not in mae_val:
                            mae = float(mae_val)
                
                # 如果关键指标没解析出来，跳过
                if mse is None or mae is None:
                    current_setting = None
                    continue

                # 2. 解析配置信息 (从 current_setting 中提取)
                # setting 格式: long term forecast ETTh2 Dropout SSSS ...
                setting_str = current_setting
                
                # --- 提取噪声类型 (放宽匹配条件，去掉下划线依赖) ---
                noise_cat = 'Clean'
                if 'Gaussian' in setting_str: noise_cat = 'Gaussian'
                elif 'Drift' in setting_str: noise_cat = 'Drift'
                elif 'Dropout' in setting_str: noise_cat = 'Dropout'
                
                # --- 提取模型名称 ---
                model_name = "Unknown"
                # 注意：如果你的模型名叫 SSSS，需要把它加到这个列表里
                # 这里去掉了下划线 _m_，直接匹配名称
                model_list = ['Autoformer', 'Crossformer', 'iTransformer', 'Pyraformer', 
                              'MICN', 'LightTS', 'TSMixer', 'SSSS'] 
                for m in model_list:
                    if m in setting_str:
                        model_name = m
                        break
                
                # --- 提取数据集 ---
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

            except ValueError:
                print(f"数值转换错误，跳过: {line[:30]}...")

            # 解析完一组数据后，清空 setting，等待下一组
            current_setting = None
            
        else:
            # 如果不包含 mse/mae，我们假设它是“配置行”
            current_setting = line

df = pd.DataFrame(data)

# 分割并保存
cats = ['Clean', 'Gaussian', 'Drift', 'Dropout']
for cat in cats:
    sub_df = df[df['Noise'] == cat].sort_values(by=['Dataset', 'Model'])
    if not sub_df.empty:
        sub_df.to_csv(f'Report_{cat}.csv', index=False)
        print(f"已生成 Report_{cat}.csv (包含 {len(sub_df)} 条记录)")
    else:
        print(f"类别 {cat} 没有数据")
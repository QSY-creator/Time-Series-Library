import os
import subprocess
import queue
import threading
import datetime
import pandas as pd
import time
import sys

# =========================================================================
# 1. 实验核心配置
# =========================================================================

PRED_LENS = [96, 192, 336, 720]

COMMON_ARGS_BASE = (
    "--task_name long_term_forecast "
    "--is_training 1 "
    "--root_path ./dataset/ " 
    "--seq_len 96 "
    "--features MS --c_out 1 " 
    "--des 'Exp' "
    "--itr 1 "            
    "--patience 3 "       
    "--train_epochs 10 " 
)

MODEL_CONFIGS = {
    'Autoformer':   {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --d_model 512 --n_heads 8", 'label_len': 48, 'batch_size': 64},
    'Crossformer':  {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --d_model 512 --n_heads 8", 'label_len': 48, 'batch_size': 32},
    'TimeMixer':    {'args': "--e_layers 2 --d_model 16 --d_ff 32 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 --learning_rate 0.01", 'label_len': 0, 'batch_size': 256},
    'iTransformer': {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --d_model 128 --d_ff 128", 'label_len': 48, 'batch_size': 64},
    'Pyraformer':   {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --d_model 512 --n_heads 8 --window_size [2,2,2]", 'label_len': 48, 'batch_size': 64},
    'LightTS':      {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7", 'label_len': 48, 'batch_size': 64},
    'MICN':         {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7", 'label_len': 96, 'batch_size': 64},
    'SSSS':         {'args': "--seg_len 24 --d_model 512 --dropout 0.5 --learning_rate 0.0001 --enc_in 7", 'label_len': 48, 'batch_size': 64}
}
MODELS = list(MODEL_CONFIGS.keys())

DATA_MAP = {
    'ETTh1':       {'code': 'ETTh1',  'file': 'ETTh1.csv',   'folder': 'ETT-small/'},
    'ETTh2':       {'code': 'ETTh2',  'file': 'ETTh2.csv',   'folder': 'ETT-small/'},
    'ETTm1':       {'code': 'ETTm1',  'file': 'ETTm1.csv',   'folder': 'ETT-small/'},
    'Traffic':     {'code': 'custom', 'file': 'traffic.csv', 'folder': 'traffic/'},
    'Electricity': {'code': 'custom', 'file': 'electricity.csv', 'folder': 'electricity/'}
}

NOISE_TYPES = {
    'Clean':    '', 
    'Gaussian': '_noise_gaussian',
    'Drift':    '_noise_drift',
    'Dropout':  '_noise_dropout'
}

LOG_FILE = "experiment_log_final.csv"

# =========================================================================
# 2. 断点续跑逻辑 & 任务装载
# =========================================================================

def get_finished_tasks():
    """读取日志文件，获取已经跑完的任务签名"""
    finished = set()
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            # 筛选状态为 Success 的任务
            # 注意：如果状态是 Failed，我们不加入集合，这样下次会重跑
            success_df = df[df['Status'] == 'Success']
            for _, row in success_df.iterrows():
                # 构造唯一签名: (Model, Dataset, Noise, PredLen)
                # 确保类型一致
                sig = (
                    str(row['Model']), 
                    str(row['Dataset']), 
                    str(row['Noise']), 
                    int(row['PredLen'])
                )
                finished.add(sig)
        except Exception as e:
            print(f"⚠️ 读取日志文件出错，将重新开始: {e}")
    return finished

def get_gpus():
    try:
        output = subprocess.check_output("nvidia-smi -L", shell=True).decode()
        count = len(output.strip().split('\n'))
        return list(range(count))
    except:
        return []

# 初始化日志
if not os.path.exists(LOG_FILE):
    df_log = pd.DataFrame(columns=['Time', 'Model', 'Dataset', 'Noise', 'PredLen', 'Status', 'GPU', 'ErrorMsg'])
    df_log.to_csv(LOG_FILE, index=False)

task_queue = queue.Queue()
finished_tasks = get_finished_tasks()
print(f"📂 检测到 {len(finished_tasks)} 个任务已完成，将自动跳过。")

# 装载任务
skipped_count = 0
for model in MODELS:
    cfg = MODEL_CONFIGS[model]
    
    for data_key, data_info in DATA_MAP.items():
        for noise_name, suffix in NOISE_TYPES.items():
            for pred_len in PRED_LENS:
                
                # 1. 检查是否已完成
                # 这里的 noise_name 要和 LOG_FILE 里的记录一致
                # LOG_FILE 里记录的是 'Gaussian', 'Clean' 等 keys
                current_sig = (model, data_key, noise_name, pred_len)
                
                if current_sig in finished_tasks:
                    skipped_count += 1
                    continue
                
                # 2. 构造文件名
                raw_filename = data_info['file']
                if suffix == '':
                    file_name = raw_filename
                else:
                    base, ext = os.path.splitext(raw_filename)
                    file_name = f"{base}{suffix}{ext}"
                
                root_path = f"./dataset/{data_info['folder']}"
                
                # 3. 构造命令
                cmd = (
                    f"python -u run.py {COMMON_ARGS_BASE} "
                    f"--model_id {data_key}_{noise_name}_{model}_pl{pred_len}_MS "
                    f"--model {model} "
                    f"--data {data_info['code']} "
                    f"--root_path {root_path} "
                    f"--data_path {file_name} "
                    f"--pred_len {pred_len} "
                    f"--label_len {cfg['label_len']} " 
                    f"--batch_size {cfg['batch_size']} "
                    f"{cfg['args']}" 
                )
                
                task_queue.put({
                    'cmd': cmd,
                    'model': model,
                    'dataset': data_key,
                    'noise': noise_name,
                    'pred_len': pred_len
                })

print(f"🚀 任务加载完毕: 跳过 {skipped_count} 个, 剩余 {task_queue.qsize()} 个待执行。")

# =========================================================================
# 3. 执行引擎 (不变)
# =========================================================================

def worker(gpu_id):
    while not task_queue.empty():
        try:
            task = task_queue.get(block=False)
        except queue.Empty:
            break
            
        full_cmd = f"{task['cmd']} --gpu {gpu_id}"
        print(f"⚡ [GPU {gpu_id}] Start: {task['model']} | {task['dataset']} | {task['noise']} | {task['pred_len']}")
        
        status = "Success"
        error_msg = ""
        start_t = time.time()
        
        try:
            env = os.environ.copy()
            ret = subprocess.run(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            if ret.returncode != 0:
                status = "Failed"
                error_msg = ret.stderr[-500:].replace('\n', ' || ')
        except Exception as e:
            status = "Error"
            error_msg = str(e)
            
        duration = round(time.time() - start_t, 2)
        
        # 写入日志 (加了 flush 确保断电也能写入)
        new_row = {
            'Time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Model': task['model'],
            'Dataset': task['dataset'],
            'Noise': task['noise'],
            'PredLen': task['pred_len'],
            'Status': status,
            'GPU': gpu_id,
            'ErrorMsg': error_msg
        }
        pd.DataFrame([new_row]).to_csv(LOG_FILE, mode='a', header=False, index=False)
        
        if status == "Success":
            print(f"✅ [GPU {gpu_id}] Done ({duration}s)")
        else:
            print(f"❌ [GPU {gpu_id}] Failed")
            
        task_queue.task_done()

gpus = get_gpus()
if not gpus:
    print("❌ 无 GPU")
    sys.exit()

print(f"💻 GPU 列表: {gpus}")
threads = []
for gpu in gpus:
    t = threading.Thread(target=worker, args=(gpu,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print("\n🎉 全部结束！")
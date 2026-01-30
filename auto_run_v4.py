import os
import subprocess
import queue
import threading
import datetime
import pandas as pd
import time
import sys
import torch

# ================= 1. 5090 专属极速配置 =================
PRED_LENS = [96, 192, 336, 720]
LOG_DIR = "./task_logs_5090"
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

csv_lock = threading.Lock()

# 基础参数
COMMON_ARGS_BASE = (
    "--task_name long_term_forecast "
    "--is_training 1 "
    "--root_path ./dataset/ " 
    "--seq_len 96 "
    "--features MS --c_out 1 " 
    "--des 'Exp' "
    "--itr 1 "            
    "--patience 2 "       
    "--train_epochs 10 "
    "--num_workers 3 "   # 注意：5090实例CPU核数少，保持为3，不要太高
)

# === 模型参数 (RTX 5090 32GB 满血版) ===
# 显存足够大，全部恢复为 256 或更高，追求最快速度
MODEL_CONFIGS = {
    'Autoformer':   {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --d_model 512 --n_heads 8", 'label_len': 48, 'batch_size': 256},
    'Crossformer':  {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --d_model 512 --n_heads 8", 'label_len': 48, 'batch_size': 128}, # Crossformer 比较吃显存，128稳妥
    'TimeMixer':    {'args': "--e_layers 2 --d_model 16 --d_ff 32 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 --learning_rate 0.01", 'label_len': 0, 'batch_size': 1024}, # 极大 Batch
    'iTransformer': {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --d_model 128 --d_ff 128", 'label_len': 48, 'batch_size': 256},
    'Pyraformer':   {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --d_model 512 --n_heads 8 --window_size [2,2,2]", 'label_len': 48, 'batch_size': 256},
    'LightTS':      {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7", 'label_len': 48, 'batch_size': 512},
    'MICN':         {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7", 'label_len': 96, 'batch_size': 256},
    'SSSS':         {'args': "--seg_len 24 --d_model 512 --dropout 0.5 --learning_rate 0.0001 --enc_in 7", 'label_len': 48, 'batch_size': 256}
}
MODELS = list(MODEL_CONFIGS.keys())

# 数据集映射 (保持不变)
DATA_MAP = {
    'ETTh1':       {'code': 'ETTh1',  'file': 'ETTh1.csv',   'folder': 'ETT-small/'},
    'ETTh2':       {'code': 'ETTh2',  'file': 'ETTh2.csv',   'folder': 'ETT-small/'},
    'ETTm1':       {'code': 'ETTm1',  'file': 'ETTm1.csv',   'folder': 'ETT-small/'},
    'Traffic':     {'code': 'custom', 'file': 'traffic.csv', 'folder': 'traffic/'},
    'Electricity': {'code': 'custom', 'file': 'electricity.csv', 'folder': 'electricity/'}
}
NOISE_TYPES = {'Clean': '', 'Gaussian': '_noise_gaussian', 'Drift': '_noise_drift', 'Dropout': '_noise_dropout'}
LOG_FILE = "experiment_log_5090.csv"

# ================= 2. 辅助函数 =================
def check_gpu_environment():
    if not torch.cuda.is_available():
        print("❌ [严重错误] PyTorch 无法调用 GPU！")
        sys.exit(1)
    count = torch.cuda.device_count()
    print(f"✅ 检测到 {count} 张 RTX 5090 显卡。")
    return list(range(count))

def initialize_log_file():
    if not os.path.exists(LOG_FILE):
        header_df = pd.DataFrame(columns=['Time', 'Model', 'Dataset', 'Noise', 'PredLen', 'Status', 'GPU', 'ErrorMsg'])
        header_df.to_csv(LOG_FILE, index=False)

def get_finished_tasks():
    finished = set()
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            for _, row in df[df['Status'] == 'Success'].iterrows():
                finished.add((str(row['Model']), str(row['Dataset']), str(row['Noise']), int(row['PredLen'])))
        except: pass
    return finished

# ================= 3. 任务装载 =================
initialize_log_file()
task_queue = queue.Queue()
finished_tasks = get_finished_tasks()

print("📦 正在装载 RTX 5090 任务队列...")
for model in MODELS:
    cfg = MODEL_CONFIGS[model]
    for data_key, data_info in DATA_MAP.items():
        for noise_name, suffix in NOISE_TYPES.items():
            for pred_len in PRED_LENS:
                if (model, data_key, noise_name, pred_len) in finished_tasks: continue
                
                raw_filename = data_info['file']
                file_name = raw_filename if suffix == '' else f"{os.path.splitext(raw_filename)[0]}{suffix}{os.path.splitext(raw_filename)[1]}"
                
                # 32GB 显存无需动态降级，直接使用配置的 Batch Size
                current_batch_size = cfg['batch_size']
                
                cmd = (
                    f"python -u run.py {COMMON_ARGS_BASE} "
                    f"--model_id {data_key}_{noise_name}_{model}_pl{pred_len}_MS "
                    f"--model {model} --data {data_info['code']} "
                    f"--root_path ./dataset/{data_info['folder']} --data_path {file_name} "
                    f"--pred_len {pred_len} --label_len {cfg['label_len']} --batch_size {current_batch_size} "
                    f"{cfg['args']}" 
                )
                
                log_path = os.path.join(LOG_DIR, f"{model}_{data_key}_{noise_name}_{pred_len}.log")
                task_queue.put({
                    'cmd': cmd, 'model': model, 'dataset': data_key, 
                    'noise': noise_name, 'pred_len': pred_len, 'log_path': log_path,
                    'batch_size': current_batch_size
                })

print(f"🚀 剩余待执行任务数: {task_queue.qsize()}")

# ================= 4. 执行引擎 =================
def worker(gpu_id):
    while not task_queue.empty():
        try: task = task_queue.get(block=False)
        except queue.Empty: break
        
        full_cmd = f"{task['cmd']} --use_gpu True --gpu 0"
        
        print(f"⚡ [GPU {gpu_id}] Start: {task['model']} | {task['dataset']} | {task['pred_len']} | BS={task['batch_size']}")
        status = "Success"
        error_msg = ""
        start_t = time.time()
        
        with open(task['log_path'], 'w') as f_log:
            try:
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                # 5090 显存大，通常不需要碎片整理，但加上也无妨
                env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                
                ret = subprocess.run(
                    full_cmd, shell=True, 
                    stdout=f_log, stderr=subprocess.STDOUT, 
                    env=env
                )
                
                # 简单的成功检查
                if ret.returncode != 0:
                    status = "Failed"
                    error_msg = "Process returned non-zero code"
                
            except Exception as e:
                status = "Error"
                error_msg = str(e)

        duration = round(time.time() - start_t, 2)
        
        new_row = {
            'Time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Model': task['model'], 'Dataset': task['dataset'],
            'Noise': task['noise'], 'PredLen': task['pred_len'],
            'Status': status, 'GPU': gpu_id, 'ErrorMsg': error_msg
        }
        
        with csv_lock:
            pd.DataFrame([new_row]).to_csv(LOG_FILE, mode='a', header=False, index=False)
        
        if status == "Success":
            print(f"✅ [GPU {gpu_id}] Done ({duration}s)")
        else:
            print(f"❌ [GPU {gpu_id}] Failed!")
            
        task_queue.task_done()

# 启动
gpus = check_gpu_environment()
threads = []
for gpu in gpus:
    t = threading.Thread(target=worker, args=(gpu,))
    t.start()
    threads.append(t)
for t in threads: t.join()
print("🎉 5090 极速训练结束")
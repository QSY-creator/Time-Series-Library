import os
import subprocess
import queue
import threading
import datetime
import pandas as pd
import time
import sys
import torch

# ================= 1. 实验配置区 =================
PRED_LENS = [96, 192, 336, 720]
LOG_DIR = "./task_logs"
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

# 基础参数 (注意: features MS, c_out 1 是你的核心需求)
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

# === 模型特有参数 (基于你上传的 .sh 文件严格校对) ===
MODEL_CONFIGS = {
    'Autoformer':   {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --d_model 512 --n_heads 8", 'label_len': 48, 'batch_size': 64},
    'Crossformer':  {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --d_model 512 --n_heads 8", 'label_len': 48, 'batch_size': 32},
    # TimeMixer: label_len=0, down_sampling_layers等参数
    'TimeMixer':    {'args': "--e_layers 2 --d_model 16 --d_ff 32 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 --learning_rate 0.01", 'label_len': 0, 'batch_size': 128},
    'iTransformer': {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --d_model 128 --d_ff 128", 'label_len': 48, 'batch_size': 64},
    # Pyraformer: window_size [2,2,2]
    'Pyraformer':   {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --d_model 512 --n_heads 8 --window_size [2,2,2]", 'label_len': 48, 'batch_size': 64},
    'LightTS':      {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7", 'label_len': 48, 'batch_size': 64},
    # MICN: label_len 96
    'MICN':         {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7", 'label_len': 96, 'batch_size': 64},
    # SSSS: seg_len 24
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
NOISE_TYPES = {'Clean': '', 'Gaussian': '_noise_gaussian', 'Drift': '_noise_drift', 'Dropout': '_noise_dropout'}
LOG_FILE = "experiment_log_final.csv"

# ================= 2. 辅助函数 =================
def check_gpu_environment():
    if not torch.cuda.is_available():
        print("❌ 致命错误: PyTorch 检测不到 GPU！请检查 Featurize 环境设置。")
        sys.exit(1)
    count = torch.cuda.device_count()
    print(f"✅ 检测到 {count} 张显卡。")
    return list(range(count))

def get_finished_tasks():
    finished = set()
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            # 只要状态是 Success，就跳过
            for _, row in df[df['Status'] == 'Success'].iterrows():
                finished.add((str(row['Model']), str(row['Dataset']), str(row['Noise']), int(row['PredLen'])))
        except: pass
    return finished

# ================= 3. 任务装载 =================
task_queue = queue.Queue()
finished_tasks = get_finished_tasks()

print("📦 正在装载任务队列...")
for model in MODELS:
    cfg = MODEL_CONFIGS[model]
    for data_key, data_info in DATA_MAP.items():
        for noise_name, suffix in NOISE_TYPES.items():
            for pred_len in PRED_LENS:
                # 跳过已完成的
                if (model, data_key, noise_name, pred_len) in finished_tasks: continue
                
                raw_filename = data_info['file']
                file_name = raw_filename if suffix == '' else f"{os.path.splitext(raw_filename)[0]}{suffix}{os.path.splitext(raw_filename)[1]}"
                
                # 构造命令
                cmd = (
                    f"python -u run.py {COMMON_ARGS_BASE} "
                    f"--model_id {data_key}_{noise_name}_{model}_pl{pred_len}_MS "
                    f"--model {model} --data {data_info['code']} "
                    f"--root_path ./dataset/{data_info['folder']} --data_path {file_name} "
                    f"--pred_len {pred_len} --label_len {cfg['label_len']} --batch_size {cfg['batch_size']} "
                    f"{cfg['args']}" 
                )
                
                log_path = os.path.join(LOG_DIR, f"{model}_{data_key}_{noise_name}_{pred_len}.log")
                
                task_queue.put({
                    'cmd': cmd, 'model': model, 'dataset': data_key, 
                    'noise': noise_name, 'pred_len': pred_len, 'log_path': log_path
                })

print(f"🚀 剩余待执行任务数: {task_queue.qsize()}")

# ================= 4. 执行引擎 (修复版) =================
def worker(gpu_id):
    while not task_queue.empty():
        try: task = task_queue.get(block=False)
        except queue.Empty: break
        
        # 【核心修复】: 永远传 --gpu 0。
        # 因为我们会用 CUDA_VISIBLE_DEVICES 把 gpu_id 伪装成第 0 张卡。
        full_cmd = f"{task['cmd']} --gpu 0"
        
        print(f"⚡ [GPU {gpu_id}] Start: {task['model']} | {task['dataset']} | {task['noise']} | {task['pred_len']}")
        status = "Success"
        error_msg = ""
        start_t = time.time()
        
        # 记录详细日志到 log 文件
        with open(task['log_path'], 'w') as f_log:
            try:
                # 【核心修复】: 这里做物理隔离
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                
                ret = subprocess.run(
                    full_cmd, shell=True, 
                    stdout=f_log, stderr=subprocess.STDOUT, 
                    env=env
                )
                
                if ret.returncode != 0:
                    status = "Failed"
                    # 读取日志最后 3 行作为简略报错
                    with open(task['log_path'], 'r') as f_err:
                        lines = f_err.readlines()
                        error_msg = "".join(lines[-3:]).replace('\n', ' ')
            except Exception as e:
                status = "Error"
                error_msg = str(e)

        duration = round(time.time() - start_t, 2)
        
        # 写入总 Log
        new_row = {
            'Time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Model': task['model'], 'Dataset': task['dataset'],
            'Noise': task['noise'], 'PredLen': task['pred_len'],
            'Status': status, 'GPU': gpu_id, 'ErrorMsg': error_msg
        }
        pd.DataFrame([new_row]).to_csv(LOG_FILE, mode='a', header=False, index=False)
        
        if status == "Success":
            print(f"✅ [GPU {gpu_id}] Done ({duration}s)")
        else:
            print(f"❌ [GPU {gpu_id}] Failed! 查看: {task['log_path']}")
            
        task_queue.task_done()

# 启动
gpus = check_gpu_environment()
threads = []
for gpu in gpus:
    t = threading.Thread(target=worker, args=(gpu,))
    t.start()
    threads.append(t)
for t in threads: t.join()
print("🎉 全部结束")
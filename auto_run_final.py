import os
import subprocess
import queue
import threading
import datetime
import pandas as pd
import time
import sys
import torch # 引入 torch 做检查

# ================= 配置区 =================
PRED_LENS = [96, 192, 336, 720]
LOG_DIR = "./task_logs" # 新增：日志文件夹
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

# 基础参数
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

# 模型配置 (保持你的原设)
MODEL_CONFIGS = {
    'Autoformer':   {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --d_model 512 --n_heads 8", 'label_len': 48, 'batch_size': 64},
    # ... (其他模型配置保持不变，为节省篇幅省略，请直接用你原来 auto_run_final.py 里的 MODEL_CONFIGS 字典)
    # 务必把剩下的 Crossformer, TimeMixer 等填回来！
    'Crossformer':  {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --d_model 512 --n_heads 8", 'label_len': 48, 'batch_size': 32},
    'TimeMixer':    {'args': "--e_layers 2 --d_model 16 --d_ff 32 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 --learning_rate 0.01", 'label_len': 0, 'batch_size': 256},
    'iTransformer': {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --d_model 128 --d_ff 128", 'label_len': 48, 'batch_size': 64},
    'Pyraformer':   {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --d_model 512 --n_heads 8 --window_size [2,2,2]", 'label_len': 48, 'batch_size': 64},
    'LightTS':      {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7", 'label_len': 48, 'batch_size': 64},
    'MICN':         {'args': "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7", 'label_len': 96, 'batch_size': 64},
    'SSSS':         {'args': "--seg_len 24 --d_model 512 --dropout 0.5 --learning_rate 0.0001 --enc_in 7", 'label_len': 48, 'batch_size': 64}
}
MODELS = list(MODEL_CONFIGS.keys())

# 数据集与噪声 (保持不变)
DATA_MAP = {
    'ETTh1':       {'code': 'ETTh1',  'file': 'ETTh1.csv',   'folder': 'ETT-small/'},
    'ETTh2':       {'code': 'ETTh2',  'file': 'ETTh2.csv',   'folder': 'ETT-small/'},
    'ETTm1':       {'code': 'ETTm1',  'file': 'ETTm1.csv',   'folder': 'ETT-small/'},
    'Traffic':     {'code': 'custom', 'file': 'traffic.csv', 'folder': 'traffic/'},
    'Electricity': {'code': 'custom', 'file': 'electricity.csv', 'folder': 'electricity/'}
}
NOISE_TYPES = {'Clean': '', 'Gaussian': '_noise_gaussian', 'Drift': '_noise_drift', 'Dropout': '_noise_dropout'}
LOG_FILE = "experiment_log_final.csv"

# ================= 核心检查 =================
def check_gpu_environment():
    print("🔍 正在检查 PyTorch GPU 环境...")
    if not torch.cuda.is_available():
        print("❌ 致命错误: PyTorch 检测不到 GPU！任务正在 CPU 上运行！")
        print("   请检查: 1. 是否安装了 cpu 版 torch? 2. 驱动是否正常?")
        sys.exit(1)
    
    count = torch.cuda.device_count()
    print(f"✅ 检测到 {count} 张显卡。准备起飞！")
    return list(range(count))

# ================= 任务装载 (带去重) =================
def get_finished_tasks():
    finished = set()
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            for _, row in df[df['Status'] == 'Success'].iterrows():
                finished.add((str(row['Model']), str(row['Dataset']), str(row['Noise']), int(row['PredLen'])))
        except: pass
    return finished

task_queue = queue.Queue()
finished_tasks = get_finished_tasks()

print("📦 装载任务中...")
for model in MODELS:
    cfg = MODEL_CONFIGS[model]
    for data_key, data_info in DATA_MAP.items():
        for noise_name, suffix in NOISE_TYPES.items():
            for pred_len in PRED_LENS:
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
                
                # 构造日志文件路径: ./task_logs/Autoformer_ETTh1_Clean_96.log
                log_path = os.path.join(LOG_DIR, f"{model}_{data_key}_{noise_name}_{pred_len}.log")
                
                task_queue.put({
                    'cmd': cmd, 'model': model, 'dataset': data_key, 
                    'noise': noise_name, 'pred_len': pred_len, 'log_path': log_path
                })

print(f"🚀 剩余任务数: {task_queue.qsize()}")

# ================= 执行引擎 (日志重定向版) =================
def worker(gpu_id):
    while not task_queue.empty():
        try: task = task_queue.get(block=False)
        except queue.Empty: break
        
        # 强制指定 GPU
        full_cmd = f"{task['cmd']} --gpu {gpu_id}"
        
        print(f"⚡ [GPU {gpu_id}] Start: {task['model']} - {task['dataset']} ({task['noise']})")
        status = "Success"
        error_msg = ""
        start_t = time.time()
        
        # === 关键修改: 输出重定向到文件，不再堵塞终端 ===
        with open(task['log_path'], 'w') as f_log:
            try:
                # 设置环境变量，确保 subprocess 只能看到这一张卡
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id) # 双重保险
                
                # 运行任务，stdout和stderr都写入文件
                ret = subprocess.run(
                    full_cmd, shell=True, 
                    stdout=f_log, stderr=subprocess.STDOUT, 
                    env=env
                )
                
                if ret.returncode != 0:
                    status = "Failed"
                    # 读取日志文件的最后几行作为错误信息
                    with open(task['log_path'], 'r') as f_err:
                        lines = f_err.readlines()
                        error_msg = "".join(lines[-5:]).replace('\n', ' ')
            except Exception as e:
                status = "Error"
                error_msg = str(e)

        duration = round(time.time() - start_t, 2)
        
        # 写入总表
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
            print(f"❌ [GPU {gpu_id}] Failed! 查看详情: {task['log_path']}")
            
        task_queue.task_done()

# 启动
gpus = check_gpu_environment() # 这里会先检查，如果没显卡直接报错退出
threads = []
for gpu in gpus:
    t = threading.Thread(target=worker, args=(gpu,))
    t.start()
    threads.append(t)
for t in threads: t.join()
print("🎉 全部结束")
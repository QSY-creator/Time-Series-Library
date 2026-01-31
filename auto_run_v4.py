import os
import subprocess
import queue
import threading
import datetime
import pandas as pd
import time
import sys
import torch

# ================= 1. 实验配置区 (4卡 5090 极限版) =================
# 预估耗时：约 5.5 - 6.0 小时
# 策略：Traffic/Electricity 求稳不崩，ETT 全速狂飙

PRED_LENS = [96, 192, 336, 720]
LOG_DIR = "./task_logs_4card"
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

csv_lock = threading.Lock()

# ⚠️ 关键设置：num_workers=2
# 你的 CPU 只有 16 核，4张卡并行时，这里绝对不能超过 2，否则 CPU 跑满会导致 GPU 闲置！
COMMON_ARGS_BASE = (
    "--task_name long_term_forecast "
    "--is_training 1 "
    "--root_path ./dataset/ " 
    "--seq_len 96 "
    "--features MS --c_out 1 " 
    "--des 'Exp' "
    "--itr 1 "            
    "--patience 2 "       # 激进早停，节省时间
    "--train_epochs 10 "
    "--num_workers 2 "    # ⚡ 绝对不要改大！
)

# ================= 调整后的模型参数配置 =================
MODEL_ARGS = {
    'Autoformer':   "--e_layers 2 --d_layers 1 --factor 3 --d_model 512 --d_ff 2048",
    'iTransformer': "--e_layers 2 --d_model 512 --d_ff 2048 --update_config",
    'Crossformer':  "--e_layers 2 --d_layers 1 --d_model 256 --d_ff 512 --seg_len 12 --factor 3",
    'LightTS':      "--e_layers 2 --d_model 128 --d_ff 256 --chunk_size 24",
    # MICN 修复：label_len 必须为 96 (与 seq_len 一致)
    'MICN':         "--e_layers 2 --d_layers 1 --factor 3 --d_model 512 --d_ff 2048 --label_len 96",
    # Pyraformer 修复：移除了错误的括号，增加了必要的参数
    'Pyraformer':   "--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7",
    # TimeMixer 修复：参考成功脚本，label_len 设为 0，添加下采样参数
    'TimeMixer':    "--e_layers 2 --d_model 16 --d_ff 32 --label_len 0 --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg",
    'SSSS':         "--e_layers 2 --d_model 256 --d_ff 512"
}
DATA_DIM = {
    'ETTh1': 7, 'ETTh2': 7, 'ETTm1': 7, 'ETTm2': 7,
    'Electricity': 321,
    'Traffic': 862
}
MODELS = list(MODEL_ARGS.keys())

DATA_MAP = {
    'ETTh1':       {'code': 'ETTh1',  'file': 'ETTh1.csv',   'folder': 'ETT-small/', 'label_len': 48},
    'ETTh2':       {'code': 'ETTh2',  'file': 'ETTh2.csv',   'folder': 'ETT-small/', 'label_len': 48},
    'ETTm1':       {'code': 'ETTm1',  'file': 'ETTm1.csv',   'folder': 'ETT-small/', 'label_len': 48},
    'Traffic':     {'code': 'custom', 'file': 'traffic.csv', 'folder': 'traffic/',   'label_len': 48},
    'Electricity': {'code': 'custom', 'file': 'electricity.csv', 'folder': 'electricity/', 'label_len': 48}
}
NOISE_TYPES = {'Clean': '', 'Gaussian': '_noise_gaussian', 'Drift': '_noise_drift', 'Dropout': '_noise_dropout'}
LOG_FILE = "experiment_log_final_4card.csv"

# ================= 2. 🧠 智能变速箱 (核心逻辑) =================
def get_optimized_batch_size(model, dataset_name, pred_len):
    """
    针对 4卡并发 + 16核CPU 的环境，计算最优 Batch Size。
    既要防止 Traffic OOM，又要防止 ETT 太慢。
    """
    # 1. 初始基准 (5090 32GB 很大，默认可以大一点)
    bs = 256 
    
    # 2. 数据集降级 (Traffic 是显存杀手)
    if dataset_name == 'Traffic': # 862 维特征
        if pred_len >= 720: return 8   # 720长度极易爆，保命要紧
        if pred_len >= 336: return 16
        return 32 # 短序列
        
    elif dataset_name == 'Electricity': # 321 维特征
        if pred_len >= 720: return 16
        if pred_len >= 336: return 32
        return 64
        
    else: # ETT 系列 (7 维特征，轻量级)
        # ETT 可以跑很快，但之前的日志显示 Autoformer 在 720/256 时爆了
        if pred_len == 720:
            return 64 # 安全水位
        if pred_len == 336:
            return 128
        return 256 # 96/192 长度全速跑
    
    # 3. 模型特殊修正
    # Crossformer 计算复杂度是 O(L^2) 且对 Channel 敏感
    if model == 'Crossformer' and dataset_name in ['Traffic', 'Electricity']:
        return 4 # 极度保守，防止卡死
        
    return bs

# ================= 3. 辅助函数 =================
def check_gpu_environment():
    if not torch.cuda.is_available():
        print("❌ [严重错误] PyTorch 无法调用 GPU！")
        sys.exit(1)
    count = torch.cuda.device_count()
    print(f"✅ 检测到 {count} 张 RTX 5090。")
    if count < 4:
        print(f"⚠️ 警告：检测到的显卡少于4张 (实际: {count})，任务将变慢。")
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

# ================= 4. 任务装载 =================
initialize_log_file()
task_queue = queue.Queue()
finished_tasks = get_finished_tasks()

# 按照“先难后易”的顺序排序，把慢任务分散开，防止最后4张卡同时跑Traffic卡死
# 但为了代码简单，这里采用交错入队
tasks_buffer = []

print("📦 正在装载任务队列...")
for model in MODELS:
    model_args = MODEL_ARGS[model]
    for data_key, data_info in DATA_MAP.items():
        for noise_name, suffix in NOISE_TYPES.items():
            for pred_len in PRED_LENS:
                if (model, data_key, noise_name, pred_len) in finished_tasks: continue
                
                raw_filename = data_info['file']
                file_name = raw_filename if suffix == '' else f"{os.path.splitext(raw_filename)[0]}{suffix}{os.path.splitext(raw_filename)[1]}"
                
                # 获取最优 Batch Size
                safe_bs = get_optimized_batch_size(model, data_key, pred_len)
                
                cmd = (
                    f"python -u run.py {COMMON_ARGS_BASE} "
                    f"--model_id {data_key}_{noise_name}_{model}_pl{pred_len}_MS "
                    f"--model {model} --data {data_info['code']} "
                    f"--root_path ./dataset/{data_info['folder']} --data_path {file_name} "
                    f"--pred_len {pred_len} --label_len {data_info['label_len']} --batch_size {safe_bs} "
                    f"{model_args}" 
                )
                
                log_path = os.path.join(LOG_DIR, f"{model}_{data_key}_{noise_name}_{pred_len}.log")
                tasks_buffer.append({
                    'cmd': cmd, 'model': model, 'dataset': data_key, 
                    'noise': noise_name, 'pred_len': pred_len, 'log_path': log_path,
                    'batch_size': safe_bs
                })

# 将任务打乱一点，避免同时跑4个Traffic导致IO瓶颈
import random
# random.shuffle(tasks_buffer) # 如果你喜欢随机可以取消注释，但顺序跑方便看进度
for t in tasks_buffer:
    task_queue.put(t)

print(f"🚀 任务装载完毕: {task_queue.qsize()} 个任务等待执行。")
print(f"⚡ 预计耗时: 6 小时左右 (请保持机器开机)")

# ================= 5. 执行引擎 =================
def worker(gpu_id):
    while not task_queue.empty():
        try: task = task_queue.get(block=False)
        except queue.Empty: break
        
        full_cmd = f"{task['cmd']} --use_gpu True --gpu 0"
        
        # 简化打印信息
        print(f"▶️ [GPU {gpu_id}] Run: {task['model']} | {task['dataset']} | L{task['pred_len']} | BS{task['batch_size']}")
        
        status = "Success"
        error_msg = ""
        start_t = time.time()
        
        with open(task['log_path'], 'w') as f_log:
            try:
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                
                ret = subprocess.run(
                    full_cmd, shell=True, 
                    stdout=f_log, stderr=subprocess.STDOUT, 
                    env=env
                )
                
                try:
                    with open(task['log_path'], 'r', encoding='utf-8', errors='ignore') as f_check:
                        log_content = f_check.read()
                        if 'mse:' in log_content.lower() and 'test shape:' in log_content.lower():
                            status = "Success"
                        elif 'out of memory' in log_content.lower():
                            status = "Failed-OOM" # 这种情况应该被上面的逻辑规避了
                            error_msg = "OOM Error"
                        elif ret.returncode != 0:
                            status = "Failed"
                            lines = log_content.split('\n')
                            error_lines = [line.strip() for line in lines[-10:] if line.strip()]
                            error_msg = str(error_lines[-1]) if error_lines else "Unknown"
                except:
                    if ret.returncode != 0: status = "Failed"

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
            # 成功时不打印太详细，保持控制台清爽
            print(f"✅ [GPU {gpu_id}] Done in {duration}s")
        else:
            print(f"❌ [GPU {gpu_id}] Failed! Msg: {error_msg}")
            
        task_queue.task_done()

# 启动
gpus = check_gpu_environment()
threads = []
for gpu in gpus:
    t = threading.Thread(target=worker, args=(gpu,))
    t.start()
    threads.append(t)
for t in threads: t.join()
print("🎉 全部任务结束")
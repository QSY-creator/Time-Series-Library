import pandas as pd
import time
import os
import sys

# 文件路径
LOG_FILE = "experiment_log_final.csv"
RESULT_FILE = "result_long_term_forecast.txt"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_experiment_stats():
    if not os.path.exists(LOG_FILE):
        return None, "日志文件尚未生成..."
    
    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty: return None, "日志为空..."
        
        total = 640 # 你大概的任务总量
        finished = len(df)
        success = len(df[df['Status'] == 'Success'])
        failed = len(df[df['Status'] != 'Success'])
        
        # 获取最近失败的记录
        recent_fails = df[df['Status'] != 'Success'].tail(5)[['Model', 'Dataset', 'Noise', 'ErrorMsg']]
        
        return {
            'total': total,
            'finished': finished,
            'success': success,
            'failed': failed,
            'df': df,
            'fails': recent_fails
        }, None
    except Exception as e:
        return None, str(e)

def get_latest_metrics():
    if not os.path.exists(RESULT_FILE):
        return []
    
    try:
        # 读取最后 5 行结果
        with open(RESULT_FILE, 'r') as f:
            lines = f.readlines()
            last_lines = lines[-5:]
        return last_lines
    except:
        return []

while True:
    clear_screen()
    stats, msg = get_experiment_stats()
    
    print("="*60)
    print(f"🚀 实验实时监控看板 (每 5s 刷新)")
    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    if stats:
        # 1. 进度统计
        progress = (stats['finished'] / stats['total']) * 100
        print(f"📊 进度: [{stats['finished']} / {stats['total']}]  |  完成率: {progress:.2f}%")
        print(f"✅ 成功: \033[92m{stats['success']}\033[0m  |  ❌ 失败: \033[91m{stats['failed']}\033[0m")
        print("-" * 60)
        
        # 2. 报错预警
        if stats['failed'] > 0:
            print("\n⚠️  [最近 5 个失败报错] (请立即检查!)")
            for _, row in stats['fails'].iterrows():
                short_err = str(row['ErrorMsg'])[:80] + "..."
                print(f"🔴 {row['Model']} on {row['Dataset']} ({row['Noise']}): {short_err}")
        else:
            print("\n✨ 目前无报错，系统运行平稳。")
            
        # 3. 性能预览
        print("-" * 60)
        print("\n📈 [最新产出的实验结果]")
        metrics = get_latest_metrics()
        for m in metrics:
            print(f"   >> {m.strip()[:100]}") # 防止太长换行
            
    else:
        print(f"等待数据中... ({msg})")
        
    time.sleep(5)
import subprocess
import os

def run_scripts(script_paths):
    success_scripts = []
    failed_scripts = {}

    # 1. 先验证当前Python进程是否能识别CUDA（排查父进程环境）
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"📌 当前Python进程CUDA可用: {cuda_available}")
        if cuda_available:
            print(f"   GPU数量: {torch.cuda.device_count()}, 当前GPU: {torch.cuda.current_device()}")
        else:
            print("⚠️ 当前Python进程未识别到CUDA，建议检查环境变量加载")
    except ImportError:
        print("⚠️ 未安装PyTorch，跳过CUDA验证")

    for script_path in script_paths:
        if not os.path.exists(script_path):
            failed_scripts[script_path] = "文件不存在"
            print(f"❌ 跳过 {script_path}: 文件不存在")
            continue

        try:
            print(f"\n🔄 正在运行 {script_path}...")
            # 2. 关键：复制当前进程的完整环境变量（包含CUDA）
            env = os.environ.copy()
            # 确保不手动覆盖CUDA_VISIBLE_DEVICES（如果需要指定GPU，可取消下面注释并修改）
            # env["CUDA_VISIBLE_DEVICES"] = "0"  # 仅当你要指定GPU时启用

            if script_path.endswith(".py"):
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=env,  # 显式传递完整环境变量
                    shell=False  # 禁用shell，避免环境变量隔离
                )
            elif script_path.endswith(".sh"):
                # 运行sh脚本时，强制用交互式bash加载完整环境
                result = subprocess.run(
                    ["bash", "-i", script_path],  # -i：交互式bash，加载~/.bashrc
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=env,  # 传递环境变量
                    shell=False
                )
            else:
                failed_scripts[script_path] = "不支持的脚本类型（仅支持.py/.sh）"
                print(f"❌ 跳过 {script_path}: 不支持的脚本类型")
                continue

            # 后续判断逻辑不变
            if result.returncode == 0:
                success_scripts.append(script_path)
                print(f"✅ {script_path} 运行成功")
            else:
                error_msg = result.stderr.strip() if result.stderr else "执行返回非0状态码"
                failed_scripts[script_path] = error_msg
                print(f"❌ {script_path} 运行失败: {error_msg[:200]}")

        except Exception as e:
            failed_scripts[script_path] = f"未知错误: {str(e)}"
            print(f"❌ 跳过 {script_path}: 未知错误 - {str(e)}")

    # 汇总报告（不变）
    print("\n" + "="*80)
    print("📊 脚本运行汇总报告")
    print("="*80)
    print(f"✅ 成功运行的脚本 ({len(success_scripts)} 个):")
    for idx, script in enumerate(success_scripts, 1):
        print(f"  {idx}. {script}")

    print(f"\n❌ 运行失败的脚本 ({len(failed_scripts)} 个):")
    if failed_scripts:
        for idx, (script, reason) in enumerate(failed_scripts.items(), 1):
            print(f"  {idx}. {script}: {reason}")
    else:
        print("  无")

# ------------------- 配置你要运行的脚本列表 -------------------
if __name__ == "__main__":
    # 替换成你自己的脚本路径（绝对路径/相对路径都可以）
    scripts_to_run = [
        "./scripts/long_term_forecast/ETT_script/DLinear_ETTh1_test.sh",
        "./scripts/long_term_forecast/ETT_script/DLinear_ETTh1_wrong.sh",
        "./scripts/long_term_forecast/ETT_script/Autoformer_ETTh1111.sh",
    ]
    # 执行批量运行
    run_scripts(scripts_to_run)


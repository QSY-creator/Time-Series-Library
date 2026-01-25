import subprocess
import os

def run_scripts(script_paths):
    """
    批量运行脚本文件，出错跳过，最后汇总结果
    
    Args:
        script_paths (list): 要运行的脚本文件路径列表
    """
    # 初始化结果记录
    success_scripts = []  # 成功的脚本
    failed_scripts = {}   # 失败的脚本: {路径: 失败原因}

    # 遍历所有脚本并执行
    for script_path in script_paths:
        # 先检查文件是否存在
        if not os.path.exists(script_path):
            failed_scripts[script_path] = "文件不存在"
            print(f"❌ 跳过 {script_path}: 文件不存在")
            continue

        try:
            print(f"\n🔄 正在运行 {script_path}...")
            # 根据脚本后缀选择执行方式（可扩展其他类型）
            if script_path.endswith(".py"):
                # 运行Python脚本
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,  # 捕获输出
                    text=True,           # 输出转为字符串
                    timeout=300          # 超时时间（5分钟），防止卡死
                )
            elif script_path.endswith(".sh"):
                # 运行Shell脚本（需确保有执行权限）
                result = subprocess.run(
                    ["bash", script_path],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
            else:
                failed_scripts[script_path] = "不支持的脚本类型（仅支持.py/.sh）"
                print(f"❌ 跳过 {script_path}: 不支持的脚本类型")
                continue

            # 判断执行是否成功（返回码为0表示成功）
            if result.returncode == 0:
                success_scripts.append(script_path)
                print(f"✅ {script_path} 运行成功")
                # 可选：打印脚本的输出
                # if result.stdout:
                #     print(f"输出: {result.stdout}")
            else:
                # 记录失败原因（包含stderr的错误信息）
                error_msg = result.stderr.strip() if result.stderr else "执行返回非0状态码"
                failed_scripts[script_path] = error_msg
                print(f"❌ {script_path} 运行失败: {error_msg[:200]}")  # 只显示前200个字符，避免过长

        except subprocess.TimeoutExpired:
            failed_scripts[script_path] = "执行超时（超过5分钟）"
            print(f"❌ 跳过 {script_path}: 执行超时")
        except PermissionError:
            failed_scripts[script_path] = "无执行权限"
            print(f"❌ 跳过 {script_path}: 无执行权限")
        except Exception as e:
            # 捕获其他未知异常
            failed_scripts[script_path] = f"未知错误: {str(e)}"
            print(f"❌ 跳过 {script_path}: 未知错误 - {str(e)}")

    # 输出最终汇总报告
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
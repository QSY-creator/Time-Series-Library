#!/bin/bash

# ================= 配置区域 =================
# 在不同服务器上，修改这个名字！
# 例如服务器1写: NODE_NAME="Server1"
# 例如服务器2写: NODE_NAME="Server2"
NODE_NAME="Server_TimesNet_wea" 

# 日志文件夹
LOG_DIR="./batch_logs"
mkdir -p "$LOG_DIR"
# ===========================================

# 1. 定义脚本列表 
# ⚠️ 注意：Bash数组元素之间用【空格】或者【换行】分隔，千万不要用逗号！
SCRIPTS=(
"./scripts/long_term_forecast/Weather_script/TimesNet.sh"


)

SUCCESS_LIST=()
FAIL_LIST=()
DATE_STR=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/${NODE_NAME}_run_${DATE_STR}.log"

# 将后续的所有输出同时打印到屏幕和日志文件
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========== 开始批量运行 [$NODE_NAME] =========="
echo "日志文件: $LOG_FILE"

for script in "${SCRIPTS[@]}"; do
    echo "--------------------------------------------------"
    echo "[$(date "+%H:%M:%S")] 正在处理: $script"

    if [ ! -f "$script" ]; then
        echo "❌ [错误] 文件不存在: $script"
        FAIL_LIST+=("$script (文件缺失)")
        continue
    fi
    
    chmod +x "$script"

    # 关键修改：动态替换脚本中的 --des 参数
    # 我们不直接运行 bash script，而是读取内容，
    # 用 sed 临时将 --des 'Exp' 替换为 --des 'Server_Name'，然后通过管道传给 bash 执行。
    # 这样你不需要手动去改那几十个 .sh 文件。
    
    echo " -> 正在以此身份运行: --des '$NODE_NAME'"
    
    # 读取脚本 -> 替换des参数 -> 执行
    cat "$script" | sed "s/--des 'Exp'/--des '$NODE_NAME'/g" | bash

    if [ $? -eq 0 ]; then
        echo "✅ [成功]: $script"
        SUCCESS_LIST+=("$script")
    else
        echo "❌ [失败]: $script"
        FAIL_LIST+=("$script")
    fi
done

echo ""
echo "========== 运行汇总 [$NODE_NAME] =========="
echo "成功: ${#SUCCESS_LIST[@]}"
echo "失败: ${#FAIL_LIST[@]}"

if [ ${#FAIL_LIST[@]} -gt 0 ]; then
    echo "失败列表:"
    for item in "${FAIL_LIST[@]}"; do echo "  - $item"; done
fi
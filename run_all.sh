#!/bin/bash

# 1. 定义你要运行的脚本列表
# 你可以手动列出，或者用 *.py, *.sh 自动获取
SCRIPTS=(
        "./scripts/long_term_forecast/ETT_script/DLinear_ETTh1.sh"
        "./scripts/long_term_forecast/ETT_script/Autoformer_ETTh1.sh"
        "./scripts/long_term_forecast/ETT_script/Crossformer_ETTh1.sh"
        "./scripts/long_term_forecast/ETT_script/TSMixer_ETTh1.sh"
        "./scripts/long_term_forecast/ETT_script/iTransformer_ETTh1.sh"
        "./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1.sh"
        "./scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh"
        "./scripts/long_term_forecast/ETT_script/FEDformer_ETTh1.sh"

        "./scripts/long_term_forecast/ETT_script/Crossformer_ETTh1_drift.sh"
        "./scripts/long_term_forecast/ETT_script/Crossformer_ETTh1_dropout.sh"
        "./scripts/long_term_forecast/ETT_script/Crossformer_ETTh1_gaussian.sh"

        "./scripts/long_term_forecast/ETT_script/TSMixer_ETTh1_dropout.sh"
        "./scripts/long_term_forecast/ETT_script/TSMixer_ETTh1_drift.sh"
        "./scripts/long_term_forecast/ETT_script/TSMixer_ETTh1_gaussian.sh"

        "./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1_drift.sh"
        "./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1_dropout.sh"
        "./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1_gaussian.sh"
        
        "./scripts/long_term_forecast/ETT_script/Autoformer_ETTh1_gaussian.sh"
        "./scripts/long_term_forecast/ETT_script/Autoformer_ETTh1_drift.sh"
        "./scripts/long_term_forecast/ETT_script/Autoformer_ETTh1_dropout.sh"

        
        "./scripts/long_term_forecast/ETT_script/DLinear_ETTh1_dropout.sh"
        "./scripts/long_term_forecast/ETT_script/DLinear_ETTh1_drift.sh"  
        "./scripts/long_term_forecast/ETT_script/DLinear_ETTh1_gaussian.sh"

        "./scripts/long_term_forecast/ETT_script/iTransformer_ETTh1_dropout.sh"
        "./scripts/long_term_forecast/ETT_script/iTransformer_ETTh1_gaussian.sh"
        "./scripts/long_term_forecast/ETT_script/iTransformer_ETTh1_drift.sh"

        "./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_gaussian.sh"
        
        "./scripts/long_term_forecast/ETT_script/PatchTST_ETTh1_drift.sh"
        "./scripts/long_term_forecast/ETT_script/PatchTST_ETTh1_dropout.sh"
        "./scripts/long_term_forecast/ETT_script/PatchTST_ETTh1_gaussian.sh"


        "./scripts/long_term_forecast/ETT_script/FEDformer_ETTh1_dropout.sh"
        "./scripts/long_term_forecast/ETT_script/FEDformer_ETTh1_drift.sh"
        "./scripts/long_term_forecast/ETT_script/FEDformer_ETTh1_gaussian.sh"

        "./scripts/long_term_forecast/ETT_script/RobustTimeMixer_ETTh1_dropout.sh"
        "./scripts/long_term_forecast/ETT_script/RobustTimeMixer_ETTh1_gaussian.sh"

        "./scripts/long_term_forecast/ETT_script/RTimeMixer2_ETTh1_dropout.sh"
        "./scripts/long_term_forecast/ETT_script/RTimeMixer2_ETTh1_gaussian.sh"
)

# 或者自动获取当前目录下所有 .py 文件：
# SCRIPTS=(*.py)

# 初始化数组来记录结果
SUCCESS_LIST=()
FAIL_LIST=()

echo "========== 开始批量运行 =========="

# 2. 循环运行
for script in "${SCRIPTS[@]}"; do
    echo "正在运行: $script ..."
    
    # 判断文件是否有执行权限，如果没有则尝试赋予，或者根据后缀调用解释器
    if [[ "$script" == *.py ]]; then
        python3 "$script"
    elif [[ "$script" == *.sh ]]; then
        bash "$script"
    else
        # 尝试直接执行（前提是有执行权限）
        ./"$script"
    fi

    # 3. 检查退出代码 ($?)
    # 0 代表成功，非 0 代表失败
    if [ $? -eq 0 ]; then
        echo "✅ [成功]: $script"
        SUCCESS_LIST+=("$script")
    else
        echo "❌ [失败]: $script (已跳过)"
        FAIL_LIST+=("$script")
    fi
    echo "----------------------------------"
done

# 4. 输出最终汇总
echo ""
echo "========== 运行汇总 =========="
echo "共运行: ${#SCRIPTS[@]} 个脚本"
echo "成功: ${#SUCCESS_LIST[@]}"
echo "失败: ${#FAIL_LIST[@]}"
echo ""

if [ ${#SUCCESS_LIST[@]} -gt 0 ]; then
    echo "成功的脚本:"
    for item in "${SUCCESS_LIST[@]}"; do echo "  - $item"; done
fi

if [ ${#FAIL_LIST[@]} -gt 0 ]; then
    echo -e "\033[31m失败的脚本:\033[0m" # 红色高亮
    for item in "${FAIL_LIST[@]}"; do echo "  - $item"; done
fi
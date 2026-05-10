export CUDA_VISIBLE_DEVICES=0,1,2

data_name=ETTm2
seq_len=96

# 将你需要跑的所有消融实验模型名称放在这个数组里
models=(
  "SSSS_wolinear"
  "SSSS_wogru"
  "SSSS_worevin"
  "SSSS_wopatching"
  "SSSS" # 你的基线模型名称，可以根据实际情况修改
)

# 外层循环遍历所有的模型
for model_name in "${models[@]}"
do
  echo "================================================================"
  echo ">>> 开始训练模型: $model_name | 数据集: $data_name"
  echo "================================================================"

  # 内层循环遍历不同的预测长度
  for pred_len in 96 192 336 720 #1440 2880 5760
  do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path $data_name.csv \
      --model_id $data_name'_'$model_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --seg_len 24 \
      --enc_in 7 \
      --d_model 512 \
      --dropout 0.5 \
      --learning_rate 0.0001 \
      --des 'Exp' \
      --itr 1
  done
done
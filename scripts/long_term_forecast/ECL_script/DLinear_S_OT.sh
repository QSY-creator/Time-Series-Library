export CUDA_VISIBLE_DEVICES=0,1,2

model_name=DLinear

for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_${pred_len} \
  --model $model_name \
  --data custom \
  --features S \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --output_base ./exp_outputs/ECL_DLinear_S_OT/
done

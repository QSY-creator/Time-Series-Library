model_name=TraditionalML
ml_model=LinearRegression

mkdir -p ./exp_outputs/ECL_${ml_model}/checkpoints

D:/Anaconda/envs/duet/python.exe -u run.py \
  --task_name traditional_ml_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --ml_model $ml_model \
  --data custom \
  --features S \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --num_workers 0 \
  --output_base ./exp_outputs/ECL_${ml_model}/

D:/Anaconda/envs/duet/python.exe -u run.py \
  --task_name traditional_ml_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --ml_model $ml_model \
  --data custom \
  --features S \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --num_workers 0 \
  --output_base ./exp_outputs/ECL_${ml_model}/

D:/Anaconda/envs/duet/python.exe -u run.py \
  --task_name traditional_ml_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --ml_model $ml_model \
  --data custom \
  --features S \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --num_workers 0 \
  --output_base ./exp_outputs/ECL_${ml_model}/

D:/Anaconda/envs/duet/python.exe -u run.py \
  --task_name traditional_ml_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --ml_model $ml_model \
  --data custom \
  --features S \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --num_workers 0 \
  --output_base ./exp_outputs/ECL_${ml_model}/

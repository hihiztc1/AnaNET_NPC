export PYTHONWARNINGS="ignore::UserWarning"
cd ..
# 执行小时级任务

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --dataset_class 'D_1248_MD' \
  --do_d 1 \
  --model AnaNET \
  --features 'S' \
  --seq_len 1248 \
  --label_len 624 \
  --pred_len 1248 \
  --size 24 \
  --major 200 \
  --e_layers 1 \
  --d_layers 1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 256 \
  --itr 1 \
  --dropout 0.05 \
  --patience 3 \
  --d_ff 1024 \
  --train_epochs 10 \
  --freq 'h'


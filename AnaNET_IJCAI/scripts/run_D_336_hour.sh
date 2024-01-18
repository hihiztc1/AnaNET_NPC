export PYTHONWARNINGS="ignore::UserWarning"
cd ..
# 执行小时级任务

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --dataset_class 'D_336_MD' \
  --do_d 1 \
  --model AnaNET \
  --features 'S' \
  --seq_len 336 \
  --label_len 168 \
  --pred_len 336 \
  --size 24 \
  --major 64 \
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


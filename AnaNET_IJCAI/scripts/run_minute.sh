export CUDA_VISIBLE_DEVICES=0

cd ..
# 执行分钟级任务
# 天
for preLen in 24
do
  labelLen=$((preLen / 2))
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --dataset_class 'app' \
    --model AnaNET \
    --features S \
    --seq_len $preLen \
    --label_len $labelLen \
    --pred_len $preLen \
    --major 16 \
    --size 6 \
    --e_layers 1 \
    --d_layers 1 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model 256 \
    --itr 1 \
    --freq 't'
done





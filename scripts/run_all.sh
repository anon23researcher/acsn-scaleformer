#!/bin/bash

# ==========================
# ACSN Reproducibility Script
# ==========================

echo "Running ACSN experiments..."

# Example: Autoformer-MS (modified dataset, H=192)
python run.py \
  --model AutoformerMS \
  --data custom \
  --root_path ./data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --features M \
  --target OT \
  --freq h \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 512 \
  --n_heads 8 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 2048 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --batch_size 32 \
  --loss mse \
  --norm_mode adaptive \
  --use_gpu True

echo "Finished."

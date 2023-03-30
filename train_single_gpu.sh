#!/bin/bash

python train.py \
    --model_name_or_path google/flan-t5-base \
    --data_path ./alpaca_data.json \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --train_batch_size 8 \
    --warmup_ratio 0.03 \
    --max_steps 10 # for testing

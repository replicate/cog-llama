#!/bin/bash

torchrun --nproc_per_node=4 --master_port=9292 train.py \
    --model_name_or_path google/flan-t5-xxl \
    --data_path ./alpaca_data.json \
    --epochs 3 \
    --learning_rate 3e-4 \
    --batch_size 8 \
    --warmup_ratio 0.03 \
    --output_path flan-t5-xxl-out \
    --peft

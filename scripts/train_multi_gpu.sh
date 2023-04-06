#!/bin/bash

torchrun --nproc_per_node=4 --master_port=9292 train.py \
    --train_data ./short_alpaca_data.json \
    --num_train_epochs 3 \
    --learning_rate 5e-4 \
    --train_batch_size 1 \
    \
    --gradient_accumulation_steps 64 \
    --logging_steps 2 \
    --warmup_ratio 0.03 
    
    # xl - batch size = 6
    # xxl
# deepspeed --num_gpus 4 --master_port=9292 train.py \
#     --train_data ./short_alpaca_data.json \
#     --num_train_epochs 3 \
#     --learning_rate 5e-4 \
#     --train_batch_size 8 \
#     --gradient_accumulation_steps 8 \
#     --logging_steps 2 \
#     --warmup_ratio 0.03 
#!/bin/bash

model_names=("flan-t5-small" "flan-t5-base" "flan-t5-large" "flan-t5-xl" "flan-t5-xxl" "flan-ul2")

for model_name in "${model_names[@]}"; do
  echo "Pushing model: $model_name"
  cog run python render_template.py --model_name $model_name
  cog login --token-stdin <<< "$COG_TOKEN"
  cog push r8.im/daanelson/$model_name
done




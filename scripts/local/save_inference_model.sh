#!/bin/bash
set -eux

python -u \
    ./save_inference_model.py \
    --model ${model} \
    --task ${task} \
    --vocab_path ${vocab_path} \
    --init_pretraining_params ${init_params} \
    --spm_model_file ${spm_model_file} \
    --inference_model_path ${init_params} \
    --config_path ${config_path}

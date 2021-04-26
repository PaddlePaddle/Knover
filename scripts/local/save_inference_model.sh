#!/bin/bash
set -eux

python -m \
    knover.scripts.save_inference_model \
    --model ${model} \
    --task ${task} \
    --vocab_path ${vocab_path} \
    --init_pretraining_params ${init_params} \
    --spm_model_file ${spm_model_file} \
    --inference_model_path ${init_params} \
    ${save_args:-} \
    --config_path ${config_path}

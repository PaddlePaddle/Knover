#!/bin/bash
set -eux

if [[ $# == 1 ]]; then
    job_conf=$1
    source ${job_conf}
fi

if [[ ${spm_model_file:-""} != "" ]]; then
    save_args="--spm_model_file ${spm_model_file} ${save_args:-}"
fi

python -m \
    knover.scripts.save_inference_model \
    --model ${model} \
    --task ${task} \
    --vocab_path ${vocab_path} \
    --init_pretraining_params ${init_params} \
    --inference_model_path ${init_params} \
    ${save_args:-} \
    --config_path ${config_path}

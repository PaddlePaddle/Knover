#!/bin/bash
set -eux

if [[ $# == 1 ]]; then
    job_conf=$1
    source ${job_conf}
elif [[ $# > 1 ]]; then
    echo "usage: sh $0 [job_conf]"
    exit -1
fi

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

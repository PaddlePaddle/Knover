#!/bin/bash
set -ux

if [[ $# == 1 ]]; then
    job_conf=$1
    source ${job_conf}
elif [[ $# > 1 ]]; then
    echo "usage: sh $0 [job_conf]"
    exit -1
fi

export FLAGS_sync_nccl_allreduce=1
export FLAGS_fuse_parameter_memory_size=64

mkdir -p ${save_path}

if [[ ${spm_model_file:-""} != "" ]]; then
    eval_args="--spm_model_file ${spm_model_file} ${eval_args:-}"
fi


python -m \
    knover.scripts.evaluate \
    --model ${model} \
    --task ${task} \
    --vocab_path ${vocab_path} \
    --config_path ${config_path} \
    --init_pretraining_params ${init_params:-""} \
    --eval_file ${eval_file} \
    --data_format ${data_format:-"raw"} \
    --file_format ${file_format:-"file"} \
    ${eval_args:-} \
    --in_tokens ${in_tokens:-"true"} \
    --batch_size ${batch_size:-8192} \
    --save_path ${save_path}
exit_code=$?

exit $exit_code

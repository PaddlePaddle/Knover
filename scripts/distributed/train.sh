#!/bin/bash

if [[ $# != 1 ]]; then
    echo "usage: sh $0 job_conf"
    exit -1
fi

job_conf=$1

export FLAGS_sync_nccl_allreduce=1
export FLAGS_fuse_parameter_memory_size=64

source ${job_conf}

mkdir -p ${save_path}

python -m \
    paddle.distributed.launch \
    --log_dir ${log_dir} \
    ./train.py \
    --is_distributed true \
    --model ${model:-"Plato"} \
    --task ${task:-"DialogGeneration"} \
    --vocab_path ${vocab_path} \
    --spm_model_file ${spm_model_file} \
    --init_pretraining_params ${init_params:-""} \
    --train_file ${train_file} \
    --valid_file ${valid_file} \
    --data_format ${data_format:-"raw"} \
    --file_format ${file_format:-"file"} \
    --config_path ${config_path} \
    --in_tokens ${in_tokens:-"false"} \
    --batch_size ${batch_size:-8192} \
    --learning_rate ${lr} \
    --use_amp ${use_amp:-"true"} \
    --use_recompute ${use_recompute:-"false"} \
    --num_epochs ${num_epochs} \
    --log_steps ${log_steps} \
    --validation_steps ${validation_steps} \
    --save_steps ${save_steps} \
    --save_path ${save_path}

#!/bin/bash

if [[ $# != 1 ]]; then
    echo "usage: sh $0 job_conf"
    exit -1
fi

job_conf=$1

export FLAGS_sync_nccl_allreduce=1
export FLAGS_fuse_parameter_memory_size=64


source ${job_conf}

mkdir -p ${save_path:-"./output"}

python -m \
    paddle.distributed.launch \
    --log_dir ${log_dir:-"./log"} \
    ./train.py \
    --is_distributed true \
    --model Plato \
    --task DialogGeneration \
    --vocab_path ./package/dialog_en/vocab.txt \
    --spm_model_file ./package/dialog_en/spm.model \
    --init_pretraining_params ${init_params} \
    --data_format ${data_format:-"raw"} \
    --file_format ${file_format:-"file"} \
    --train_file ${train_file} \
    --valid_file ${valid_file} \
    --in_tokens ${in_tokens} \
    --batch_size ${batch_size} \
    --learning_rate ${lr} \
    --use_amp ${use_amp} \
    --use_recompute ${use_recompute} \
    --num_epochs ${num_epochs} \
    --skip_steps ${skip_steps} \
    --save_steps ${save_steps} \
    --validation_steps ${validation_steps} \
    --save_path ${save_path:-"./output"} \
    --config_path ${config_path}

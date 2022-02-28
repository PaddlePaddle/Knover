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

if [[ ${log_dir:-""} != "" ]]; then
    mkdir -p ${log_dir}
    distributed_args="${distributed_args:-} --log_dir ${log_dir}"
fi

if [[ ${use_sharding:-"false"} == "true" ]]; then
    export FLAGS_eager_delete_tensor_gb=3.0
fi

fleetrun \
    ${distributed_args:-} \
    ./knover/scripts/evaluate.py \
    --is_distributed true \
    --model ${model:-"Plato"} \
    --task ${task:-"DialogGeneration"} \
    --vocab_path ${vocab_path} \
    --init_pretraining_params ${init_params:-""} \
    --eval_file ${eval_file} \
    --data_format ${data_format:-"raw"} \
    --file_format ${file_format:-"file"} \
    --config_path ${config_path} \
    ${eval_args:-} \
    --in_tokens ${in_tokens:-"true"} \
    --batch_size ${batch_size:-8192} \
    --use_sharding ${use_sharding:-"false"} \
    --save_path ${save_path}
exit_code=$?

if [[ $exit_code != 0 ]]; then
    rm ${save_path}/*.finish
fi

exit $exit_code

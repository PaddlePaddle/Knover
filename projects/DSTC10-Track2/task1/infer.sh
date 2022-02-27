#!/bin/bash
set -ux

cd ../../..

source ./projects/DSTC10-Track2/task1/conf/infer.conf
mkdir -p ${log_dir}
mkdir -p ${save_path}

export FLAGS_sync_nccl_allreduce=1
export FLAGS_fuse_parameter_memory_size=64
export CUDA_VISIBLE_DEVICES=0

python -u ./projects/DSTC10-Track2/task1/infer.py \
    --model ${model:-"UnifiedTransformer"} \
    --vocab_path ${vocab_path} \
    --specials_path ${specials_path:-""} \
    --do_lower_case ${do_lower_case:-"false"} \
    --spm_model_file ${spm_model_file} \
    --init_pretraining_params ${init_params:-""} \
    --infer_file ${infer_file} \
    --db_file ${db_file} \
    --session_to_sample_mapping_file ${session_to_sample_mapping_file} \
    --dial_batch_size ${dial_batch_size:-8} \
    --normalization ${normalization:-"false"} \
    --db_guidance ${db_guidance:-"false"} \
    --data_format ${data_format:-"raw"} \
    --file_format ${file_format:-"file"} \
    --config_path ${config_path} \
    ${infer_args:-} \
    --save_path ${save_path} >> "${log_dir}/workerlog.0"

#!/bin/bash
set -eux

cd ../..

export DATASET="multiwoz"
# export DATASET="woz"
export DATA_PATH="./projects/AG-DST/data"
export MODEL_PATH="./projects/AG-DST/models"
export OUTPUT_PATH="./projects/AG-DST/output"

source ./projects/AG-DST/conf/infer.conf
mkdir -p ${log_dir}
mkdir -p ${save_path}

# infer
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fuse_parameter_memory_size=64
export CUDA_VISIBLE_DEVICES=0
python -u ./projects/AG-DST/infer_dst.py \
    --model ${model:-"UnifiedTransformer"} \
    --task ${task:-"DialogGeneration"} \
    --vocab_path ${vocab_path} \
    --specials_path ${specials_path:-""} \
    --spm_model_file ${spm_model_file} \
    --init_pretraining_params ${init_params:-""} \
    --infer_file ${infer_file} \
    --data_format ${data_format:-"raw"} \
    --file_format ${file_format:-"file"} \
    --config_path ${config_path} \
    --dataset ${dataset} \
    --dial_batch_size ${dial_batch_size:-32} \
    ${infer_args:-} \
    --save_path ${save_path} >> "${log_dir}/workerlog.0"

# evaluate
python ./projects/AG-DST/evaluate_dst.py \
    --inference_labels "${save_path}/inference_labels.json" \
    --ground_labels "${DATA_PATH}/${DATASET}/processed/test_labels.json" \
    --dataset ${DATASET} \
    --save_path ${save_path}

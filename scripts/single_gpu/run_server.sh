#!/bin/bash
set -eux

if [[ $# == 1 ]]; then
    job_conf=$1
    source ${job_conf}
elif [[ $# > 1 ]]; then
    echo "usage: sh $0 [job_conf]"
    exit -1
fi

export FLAGS_sync_nccl_allreduce=1
export FLAGS_fuse_parameter_memory_size=64

if [[ ${spm_model_file:-""} != "" ]]; then
    save_args="--spm_model_file ${spm_model_file} ${save_args:-}"
    infer_args="--spm_model_file ${spm_model_file} ${infer_args:-}"
fi

# Process NSP model(for reranking in dialogue generation task).
if [[ ${nsp_init_params:-} != "" ]]; then
    if [[ ! -e "${nsp_init_params}/__model__" ]]; then
        python -m \
            knover.scripts.save_inference_model \
            --model NSPModel \
            --task NextSentencePrediction \
            --vocab_path ${vocab_path} \
            --init_pretraining_params ${nsp_init_params} \
            --inference_model_path ${nsp_init_params} \
            ${save_args:-} \
            --config_path ${config_path}
    fi
    infer_args="--nsp_inference_model_path ${nsp_init_params} ${infer_args:-}"
fi

if [[ $infer_args =~ "--use_amp true" ]]; then
    if [[ ! -d ${init_params}-fp16 ]]; then
        python ./knover/tools/convert_checkpoint.py \
            --param_path ${init_params} \
            --save_path ${init_params}-fp16 \
            --convert_type fp16
    fi
    init_params="${init_params}-fp16"
fi

python -m \
    knover.scripts.run_server \
    --api_name ${api_name:-"chitchat"} \
    --bot_name ${bot_name} \
    --port ${port:-8088} \
    --model ${model} \
    --vocab_path ${vocab_path} \
    --config_path ${config_path} \
    --init_pretraining_params ${init_params} \
    ${infer_args:-}

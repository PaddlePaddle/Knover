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

if [[ ${log_dir:-""} != "" ]]; then
    distributed_args="${distributed_args:-} --log_dir ${log_dir}"
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
            --spm_model_file ${spm_model_file} \
            --inference_model_path ${nsp_init_params} \
            ${save_args:-} \
            --config_path ${config_path}
    fi
    infer_args="--nsp_inference_model_path ${nsp_init_params} ${infer_args:-}"
fi

if [[ $infer_args =~ "--use_sharding true" ]]; then
    if [[ ${CUDA_VISIBLE_DEVICES:-""} != "" ]]; then
        CUDA_VISIBLE_DEVICE_ARRAY=(${CUDA_VISIBLE_DEVICES//,/ })
        MP_DEGREE=${#CUDA_VISIBLE_DEVICE_ARRAY[@]}
    else
        MP_DEGREE=${cards:-$(nvidia-smi -L | wc -l)}
    fi
    infer_args="$infer_args --mp_degree $MP_DEGREE"
    if [[ ! -d ${init_params}-mp${MP_DEGREE} ]]; then
        python ./knover/tools/split_checkpoint.py \
            --param_path ${init_params} \
            --save_path ${init_params}-mp${MP_DEGREE} \
            --num_partitions ${MP_DEGREE}
    fi
    init_params="${init_params}-mp${MP_DEGREE}"
fi

fleetrun \
    ${distributed_args:-} \
    ./knover/scripts/infer.py \
    --is_distributed true \
    --model ${model} \
    --task ${task} \
    --vocab_path ${vocab_path} \
    --config_path ${config_path} \
    --do_lower_case ${do_lower_case:-"false"} \
    --spm_model_file ${spm_model_file} \
    --init_pretraining_params ${init_params} \
    --infer_file ${infer_file} \
    --data_format ${data_format:-"raw"} \
    --file_format ${file_format:-"file"} \
    --output_name ${output_name} \
    ${infer_args:-} \
    --in_tokens ${in_tokens:-"false"} \
    --batch_size ${batch_size:-1} \
    --save_path ${save_path}
exit_code=$?

if [[ $exit_code != 0 ]]; then
    rm ${save_path}/*.finish
fi

exit $exit_code

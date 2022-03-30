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


python -m \
    paddle.distributed.launch \
    ${distributed_args:-} \
    ./projects/PLATO-KAG/tools/get_dense_emb.py \
    --is_distributed true \
    --model ${model:-"PlatoKAG"} \
    --task ${task:-"DenseEmbedding"} \
    --vocab_path ${vocab_path} \
    --specials_path ${specials_path:-""} \
    --do_lower_case ${do_lower_case:-"false"} \
    --spm_model_file ${spm_model_file} \
    --init_pretraining_params ${init_params:-""} \
    --infer_file ${src_infer_file} \
    --data_format ${data_format:-"tokenized"} \
    --file_format ${file_format:-"file"} \
    --config_path ${config_path} \
    --embedding_type src \
    --save_name ${src_save_name} \
    ${infer_args:-} \
    --in_tokens ${in_tokens:-"true"} \
    --batch_size ${batch_size:-1} \
    --save_path ${save_path}
exit_code=$?

if [[ $exit_code != 0 ]]; then
    rm ${save_path}/*.finish
fi

python -m \
    paddle.distributed.launch \
    ${distributed_args:-} \
    ./projects/PLATO-KAG/tools/get_dense_emb.py \
    --is_distributed true \
    --model ${model:-"PlatoKAG"} \
    --task ${task:-"DenseEmbedding"} \
    --vocab_path ${vocab_path} \
    --specials_path ${specials_path:-""} \
    --do_lower_case ${do_lower_case:-"false"} \
    --spm_model_file ${spm_model_file} \
    --init_pretraining_params ${init_params:-""} \
    --infer_file ${knowledge_infer_file} \
    --data_format ${data_format:-"tokenized"} \
    --file_format ${file_format:-"file"} \
    --config_path ${config_path} \
    --embedding_type knowledge \
    --save_name ${knowledge_save_name} \
    ${infer_args:-} \
    --in_tokens ${in_tokens:-"true"} \
    --batch_size ${batch_size:-1} \
    --save_path ${save_path}
exit_code=$?

if [[ $exit_code != 0 ]]; then
    rm ${save_path}/*.finish
fi

exit $exit_code


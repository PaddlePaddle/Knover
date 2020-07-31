#!/bin/bash
set -eux

export CUDA_VISIBLE_DEVICES=2

# change to Knover working directory
SCRIPT=`realpath "$0"`
KNOVER_DIR=`dirname ${SCRIPT}`/../..
cd $KNOVER_DIR

model_size=32L

export vocab_path=./package/dialog_en/vocab.txt
export spm_model_file=./package/dialog_en/spm.model
export config_path=./package/dialog_en/plato/${model_size}.json

nsp_init_params=./${model_size}/NSP
if [ ! -e "${nsp_init_params}/__model__" ]; then
    export model=NSPModel
    export task=NextSentencePrediction
    export init_params=$nsp_init_params
    ./scripts/local/save_inference_model.sh
fi

export model=Plato
export init_params=./${model_size}/Plato
export infer_file=./data/dailydialog_test_60.tsv
export output_name="response"
export infer_args="\
    --do_generation true \
    --nsp_inference_model_path ${nsp_init_params} \
    --ranking_score nsp_score \
    --do_lower_case true \
    --batch_size 10 \
    --mem_efficient true"

python -u \
    ./interaction.py \
    --model ${model} \
    --vocab_path ${vocab_path} \
    --init_pretraining_params ${init_params} \
    --spm_model_file ${spm_model_file} \
    $infer_args \
    --config_path ${config_path}

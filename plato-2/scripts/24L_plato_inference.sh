#!/bin/bash
set -eux

export CUDA_VISIBLE_DEVICES=0

INFER_FILE=./data/dailydialog_test_60.tsv
SAVE_PATH=./output

SCRIPT=`realpath "$0"`
KNOVER_DIR=`dirname ${SCRIPT}`/../..
cd $KNOVER_DIR

MODEL_SIZE=24L

mkdir -p ${SAVE_PATH}

if [ ! -e "${MODEL_SIZE}/NSP/__model__" ]; then
    sh scripts/local/save_nsp_model.sh ${MODEL_SIZE}
fi

python -u \
    ./infer.py \
    --model Plato \
    --task DialogGeneration \
    --vocab_path ./package/dialog_en/vocab.txt \
    --do_lower_case false \
    --init_pretraining_params ./${MODEL_SIZE}/Plato \
    --spm_model_file ./package/dialog_en/spm.model \
    --infer_file $INFER_FILE \
    --output_name response \
    --save_path ${SAVE_PATH} \
    --nsp_inference_model_path ./${MODEL_SIZE}/NSP \
    --ranking_score nsp_score \
    --do_generation true \
    --batch_size 10 \
    --config_path ./package/dialog_en/plato/${MODEL_SIZE}.json

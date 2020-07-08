#!/bin/bash
set eux

export CUDA_VISIBLE_DEVICES=5

INFER_FILE=./data/dailydialog_test_60.tsv

mkdir -p ./output

python -u \
    ./infer.py \
    --model Plato \
    --task DialogGeneration \
    --vocab_path ./package/dialog_en/vocab.txt \
    --do_lower_case false \
    --init_pretraining_params ./24L/Plato \
    --spm_model_file ./package/dialog_en/spm.model \
    --infer_file $INFER_FILE \
    --output_name response \
    --save_path ./output \
    --nsp_inference_model_path ./24L/NSP \
    --ranking_score nsp_score \
    --do_generation true \
    --batch_size 10 \
    --config_path ./package/dialog_en/plato/24L.json

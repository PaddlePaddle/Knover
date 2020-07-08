#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

MODEL_SIZE=$1

python -u \
    ./save_nsp_model.py \
    --model NSPModel \
    --task NextSentencePrediction \
    --vocab_path ./package/dialog_en/vocab.txt \
    --init_pretraining_params ./${MODEL_SIZE}/NSP \
    --spm_model_file ./package/dialog_en/spm.model \
    --nsp_inference_model_path ./${MODEL_SIZE}/NSP \
    --config_path ./package/dialog_en/plato/${MODEL_SIZE}.json

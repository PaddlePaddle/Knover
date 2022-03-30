#!/bin/bash
set -ex

cd ../../..
export DATA_PATH="$PWD/projects/PLATO-KAG/holle/data"
export MODEL_PATH="$PWD/projects/PLATO-KAG/holle/models"
export OUTPUT_PATH="$PWD/projects/PLATO-KAG/holle/output"

# preprocess dataset
python ./projects/PLATO-KAG/holle/build_training_data.py \
    --out_file $DATA_PATH/kag_train.tsv \
    --data_file $DATA_PATH/train_data_preprocessed.json \
    --num_epochs 2 \
    --do_lower \
    --max_knowledge 32

python ./projects/PLATO-KAG/holle/build_training_data.py \
    --out_file $DATA_PATH/kag_dev.tsv \
    --data_file $DATA_PATH/dev_data_preprocessed.json \
    --num_epochs 2 \
    --do_lower \
    --max_knowledge 32

# pre-tokenize to optimize the training speed
export PYTHONPATH=.
python ./knover/tools/pre_tokenize.py \
    --vocab_path ./projects/PLATO-KAG/vocab.txt \
    --specials_path ./projects/PLATO-KAG/specials.txt \
    --spm_model_file ./package/dialog_en/spm.model \
    --input_file $DATA_PATH/kag_train.tsv \
    --output_file $DATA_PATH/kag_train.tokenized.tsv

python ./knover/tools/pre_tokenize.py \
    --vocab_path ./projects/PLATO-KAG/vocab.txt \
    --specials_path ./projects/PLATO-KAG/specials.txt \
    --spm_model_file ./package/dialog_en/spm.model \
    --input_file $DATA_PATH/kag_dev.tsv \
    --output_file $DATA_PATH/kag_dev.tokenized.tsv

# train
./scripts/local/job.sh ./projects/PLATO-KAG/holle/train.conf
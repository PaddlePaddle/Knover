#!/bin/bash
set -ex

cd ../../..
export DATA_PATH="$PWD/projects/PLATO-KAG/wow/data"
export MODEL_PATH="$PWD/projects/PLATO-KAG/wow/models"
export OUTPUT_PATH="$PWD/projects/PLATO-KAG/wow/output"

# preprocess dataset
python ./projects/PLATO-KAG/wow/build_training_data.py \
    --out_file $DATA_PATH/kag_train.tsv \
    --data_file $DATA_PATH/train.json \
    --num_epochs 2 \
    --do_lower \
    --max_knowledge 32

python ./projects/PLATO-KAG/wow/build_training_data.py \
    --out_file $DATA_PATH/kag_valid_random_split.tsv \
    --data_file $DATA_PATH/valid_random_split.json \
    --num_epochs 1 \
    --do_lower \
    --max_knowledge 32

python ./projects/PLATO-KAG/wow/build_training_data.py \
    --out_file $DATA_PATH/kag_valid_topic_split.tsv \
    --data_file $DATA_PATH/valid_topic_split.json \
    --num_epochs 1 \
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
    --input_file $DATA_PATH/kag_valid_random_split.tsv \
    --output_file $DATA_PATH/kag_valid_random_split.tokenized.tsv

python ./knover/tools/pre_tokenize.py \
    --vocab_path ./projects/PLATO-KAG/vocab.txt \
    --specials_path ./projects/PLATO-KAG/specials.txt \
    --spm_model_file ./package/dialog_en/spm.model \
    --input_file $DATA_PATH/kag_valid_topic_split.tsv \
    --output_file $DATA_PATH/kag_valid_topic_split.tokenized.tsv

# train
./scripts/local/job.sh ./projects/PLATO-KAG/wow/train.conf
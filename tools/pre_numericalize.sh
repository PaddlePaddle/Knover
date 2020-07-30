#!/bin/bash

python ./tools/pre_numericalize.py \
    --vocab_path ./package/dialog_en/vocab.txt \
    --specials_path ./package/dialog_en/specials.txt \
    --spm_model_file ./package/dialog_en/spm.model \
    --input_file ./data/train.tsv \
    --output_file ./data/train.numerical.tsv

#!/bin/bash

python -m \
    knover.tools.pre_numericalize \
    --vocab_path ./package/dialog_en/vocab.txt \
    --spm_model_file ./package/dialog_en/spm.model \
    --input_file ./data/example/train.tsv \
    --output_file ./data/example/train.numerical.tsv

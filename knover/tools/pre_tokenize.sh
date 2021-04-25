#!/bin/bash

python -m \
    knover.tools.pre_tokenize \
    --vocab_path ./package/dialog_en/vocab.txt \
    --specials_path ./package/dialog_en/specials.txt \
    --spm_model_file ./package/dialog_en/spm.model \
    --input_file ./data/example/train.tsv \
    --output_file ./data/example/train.tokenized.tsv

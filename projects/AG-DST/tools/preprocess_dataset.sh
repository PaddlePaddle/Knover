#!/bin/bash
set -eux

export PYTHONPATH=$(dirname "$0")/..:${PYTHONPATH:-}

for dataset in "multiwoz" "woz"
do
    python ./projects/AG-DST/tools/preprocess_dataset.py \
        --data_path "./projects/AG-DST/data/${dataset}/" \
        --save_path "./projects/AG-DST/data/${dataset}/processed/" \
        --mapping_file "./projects/AG-DST/data/mapping.pair"
done

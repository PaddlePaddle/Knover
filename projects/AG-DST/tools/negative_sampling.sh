#!/bin/bash
set -eux

export PYTHONPATH=$(dirname "$0")/..:${PYTHONPATH:-}

for dataset in "multiwoz" "woz"
do
    for data_type in "train" "dev"
    do
        python ./projects/AG-DST/tools/negative_sampling.py \
            --data_file "./projects/AG-DST/data/${dataset}/processed/${data_type}_data.json" \
            --db_path "./projects/AG-DST/data/${dataset}/db/" \
            --out_file "./projects/AG-DST/data/${dataset}/processed/${data_type}_data_withneg.json" \
            --dataset "${dataset}"
    done
done

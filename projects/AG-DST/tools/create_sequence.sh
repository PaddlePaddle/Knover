#!/bin/bash
set -eux

export PYTHONPATH=$(dirname "$0")/..:${PYTHONPATH:-}

for dataset in "multiwoz" "woz"
do
    for data_type in "train" "dev"
    do
        python ./projects/AG-DST/tools/create_sequence.py \
            --data_file "./projects/AG-DST/data/${dataset}/processed/${data_type}_data_withneg.json" \
            --save_path "./projects/AG-DST/data/${dataset}/processed/" \
            --dataset "${dataset}" \
            --data_type "${data_type}"
    done

    python ./projects/AG-DST/tools/create_sequence.py \
        --data_file "./projects/AG-DST/data/${dataset}/processed/test_data.json" \
        --save_path "./projects/AG-DST/data/${dataset}/processed/" \
        --dataset "${dataset}" \
        --data_type "test"
done
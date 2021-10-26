#!/bin/bash
set -eux

cd ../..

export DATASET="multiwoz"
# export DATASET="woz"
export DATA_PATH="./projects/AG-DST/data"
export MODEL_PATH="./projects/AG-DST/models"
export OUTPUT_PATH="./projects/AG-DST/output"

# train
export CUDA_VISIBLE_DEVICES=0,1,2,3
./scripts/local/job.sh ./projects/AG-DST/conf/train_${DATASET}.conf

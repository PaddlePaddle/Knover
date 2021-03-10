#!/bin/bash
set -ex

# generate dataset
python ./projects/DSTC9-Track1/task1/generate_inference_dataset_with_context.py \
    --log_file ${DATA_PATH}/${DATASET_TYPE}_logs.json \
    --out_file ${DATA_PATH}/task1_${DATASET_TYPE}.tsv

# run inference
./scripts/local/job.sh ./projects/DSTC9-Track1/task1/infer_with_context.conf

# convert to output format
python ./projects/DSTC9-Track1/task1/post_process_inference_result_with_context.py \
    --pred_file ${OUTPUT_PATH}/task1_${DATASET_TYPE}/output/inference_output.txt \
    --out_file ${OUTPUT_PATH}/task1_${DATASET_TYPE}.output.json

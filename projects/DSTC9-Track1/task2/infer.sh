#!/bin/bash
set -ex

# generate dataset
python projects/DSTC9-Track1/task2/generate_inference_dataset.py \
    --in_file ${OUTPUT_PATH}/task1_${DATASET_TYPE}.output.json \
    --out_file ${DATA_PATH}/task2_${DATASET_TYPE}.tsv \
    --log_file ${DATA_PATH}/${DATASET_TYPE}_logs.json \
    --knowledge_file ${DATA_PATH}/${DATASET_TYPE}_knowledge.json \
    --do_lowercase

# run inference
./scripts/local/job.sh ./projects/DSTC9-Track1/task2/infer.conf

# post-process inference result
python projects/DSTC9-Track1/task2/post_process_inference_result.py \
    --in_file ${OUTPUT_PATH}/task1_${DATASET_TYPE}.output.json \
    --pred_file ${OUTPUT_PATH}/task2_${DATASET_TYPE}/output/inference_output.txt \
    --out_file ${OUTPUT_PATH}/task2_${DATASET_TYPE}.output.json \
    --knowledge_file ${DATA_PATH}/${DATASET_TYPE}_knowledge.json

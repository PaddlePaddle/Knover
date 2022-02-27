#!/bin/bash
set -ex

# generate dataset
python -u \
    ./projects/DSTC10-Track2/task2/subtask3/generate_inference_dataset.py \
    --in_file ${OUTPUT_PATH}/subtask2_${DATASET_TYPE}.output.json \
    --out_file ${DATA_PATH}/subtask3_${DATASET_TYPE}.tsv \
    --log_file ${DATA_PATH}/${DATASET_TYPE}_minimal_logs.json \
    --knowledge_file ${DATA_PATH}/minimal_knowledge.json

# run inference
./scripts/local/job.sh ./projects/DSTC10-Track2/task2/subtask3/infer.conf

# post-process inference result
python ./projects/DSTC10-Track2/task2/subtask3/post_process_inference_result.py \
    --in_file ${OUTPUT_PATH}/subtask2_${DATASET_TYPE}.output.json \
    --pred_file ${OUTPUT_PATH}/subtask3_${DATASET_TYPE}/output/inference_output.txt \
    --out_file ${OUTPUT_PATH}/subtask3_${DATASET_TYPE}.output.json

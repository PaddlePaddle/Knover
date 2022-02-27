#!/bin/bash
set -ex

export PYTHONPATH=.
# generate dataset
python ./projects/DSTC10-Track2/task2/subtask1/generate_inference_dataset.py \
    --log_file ${DATA_PATH}/${DATASET_TYPE}_marked_logs.json \
    --out_file ${DATA_PATH}/subtask1_${DATASET_TYPE}.tsv

# numericalize inputs
python ./projects/DSTC10-Track2/task2/subtask1/pre_numericalize.py \
    --vocab_path ./projects/DSTC10-Track2/task2/vocab.txt \
    --specials_path ./projects/DSTC10-Track2/task2/specials.txt \
    --spm_model_file ./package/dialog_en/spm.model \
    --use_role true \
    --position_style relative \
    --input_file ${DATA_PATH}/subtask1_${DATASET_TYPE}.tsv \
    --output_file ${DATA_PATH}/subtask1_${DATASET_TYPE}.numerical.tsv

# run inference
./scripts/local/job.sh ./projects/DSTC10-Track2/task2/subtask1/infer.conf

# convert to output format
python ./projects/DSTC10-Track2/task2/subtask1/post_process_inference_result.py \
    --pred_file ${OUTPUT_PATH}/subtask1_${DATASET_TYPE}/output/inference_output.txt \
    --out_file ${OUTPUT_PATH}/subtask1_${DATASET_TYPE}.output.json

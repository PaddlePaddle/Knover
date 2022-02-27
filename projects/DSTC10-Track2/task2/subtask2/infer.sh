#!/bin/bash
set -ex

export PYTHONPATH=.
# generate dataset
python ./projects/DSTC10-Track2/task2/subtask2/generate_inference_dataset.py \
    --in_file ${OUTPUT_PATH}/subtask1_${DATASET_TYPE}.output.json \
    --out_file ${DATA_PATH}/subtask2_${DATASET_TYPE}.tsv \
    --log_file ${DATA_PATH}/${DATASET_TYPE}_marked_logs.json \
    --knowledge_file ${DATA_PATH}/minimal_knowledge.json \
    --do_lowercase

# numericalize inputs
python ./projects/DSTC10-Track2/task2/subtask2/pre_numericalize.py \
    --vocab_path ./projects/DSTC10-Track2/task2/vocab.txt \
    --specials_path ./projects/DSTC10-Track2/task2/specials.txt \
    --spm_model_file ./package/dialog_en/spm.model \
    --position_style relative \
    --use_role true \
    --max_src_len 384 \
    --max_seq_len 512 \
    --input_file ${DATA_PATH}/subtask2_${DATASET_TYPE}.tsv \
    --output_file ${DATA_PATH}/subtask2_${DATASET_TYPE}.numerical.tsv

# run inference
./scripts/local/job.sh ./projects/DSTC10-Track2/task2/subtask2/infer.conf

# # post-process inference result
python ./projects/DSTC10-Track2/task2/subtask2/post_process_inference_result.py \
    --in_file ${OUTPUT_PATH}/subtask1_${DATASET_TYPE}.output.json \
    --pred_file ${OUTPUT_PATH}/subtask2_${DATASET_TYPE}/output/inference_output.txt \
    --out_file ${OUTPUT_PATH}/subtask2_${DATASET_TYPE}.output.json \
    --knowledge_file ${DATA_PATH}/minimal_knowledge.json

#!/bin/bash
set -ex

cd ../../..
export DATA_PATH="$PWD/projects/PLATO-KAG/wow/data"
export MODEL_PATH="$PWD/projects/PLATO-KAG/wow/models"
export OUTPUT_PATH="$PWD/projects/PLATO-KAG/wow/output"
# phase: test / valid
export PHASE="test"
# split: random for "seen" / topic for "unseen"
export SPLIT="random"

# build infer data for selection
python ./projects/PLATO-KAG/wow/build_selection_infer_data.py \
    --out_src_file $DATA_PATH/${PHASE}_${SPLIT}_split_src.tsv \
    --out_doc_file $DATA_PATH/${PHASE}_${SPLIT}_split_doc.tsv \
    --out_desc_file $DATA_PATH/${PHASE}_${SPLIT}_split_desc.json \
    --data_file $DATA_PATH/${PHASE}_${SPLIT}_split.json \
    --do_lower

# infer selection
./scripts/local/job.sh ./projects/PLATO-KAG/wow/infer_selection.conf

# eval selection and build infer data for generation
python ./projects/PLATO-KAG/wow/build_generation_infer_data.py \
    --data_file $DATA_PATH/${PHASE}_${SPLIT}_split.json \
    --doc_file $DATA_PATH/${PHASE}_${SPLIT}_split_doc.tsv \
    --desc_file $DATA_PATH/${PHASE}_${SPLIT}_split_desc.json \
    --input_folder $OUTPUT_PATH/${PHASE}_${SPLIT}/selection/output \
    --output_folder $OUTPUT_PATH/${PHASE}_${SPLIT}/selection/output \
    --do_lower

# eval ppl
./scripts/local/job.sh ./projects/PLATO-KAG/wow/eval_generation.conf

# infer generation
./scripts/local/job.sh ./projects/PLATO-KAG/wow/infer_generation.conf

# eval F1
python ./projects/PLATO-KAG/tools/generation_metrics.py \
    --refer_file $OUTPUT_PATH/${PHASE}_${SPLIT}/selection/output/infer_data.tsv \
    --hypo_file ${OUTPUT_PATH}/${PHASE}_${SPLIT}/generation/output/inference_output.txt



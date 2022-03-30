#!/bin/bash
set -ex

cd ../../..
export DATA_PATH="$PWD/projects/PLATO-KAG/holle/data"
export MODEL_PATH="$PWD/projects/PLATO-KAG/holle/models"
export OUTPUT_PATH="$PWD/projects/PLATO-KAG/holle/output"

# build infer data for selection
python ./projects/PLATO-KAG/holle/build_selection_infer_data.py \
    --out_src_file $DATA_PATH/test_src.tsv \
    --out_doc_file $DATA_PATH/test_doc.tsv \
    --out_desc_file $DATA_PATH/test_desc.json \
    --multi_ref_file $DATA_PATH/multi_reference_test.json \
    --data_file $DATA_PATH/test_data_preprocessed.json \
    --do_lower

# infer selection
./scripts/local/job.sh ./projects/PLATO-KAG/holle/infer_selection.conf

# eval selection and build infer data for generation
python ./projects/PLATO-KAG/holle/build_generation_infer_data.py \
    --data_file $DATA_PATH/test_data_preprocessed.json \
    --doc_file $DATA_PATH/test_doc.tsv \
    --desc_file $DATA_PATH/test_desc.json \
    --multi_ref_file $DATA_PATH/multi_reference_test.json \
    --input_folder ${OUTPUT_PATH}/test/selection/output \
    --output_folder ${OUTPUT_PATH}/test/selection/output \
    --do_lower

# single
# eval ppl
./scripts/local/job.sh ./projects/PLATO-KAG/holle/eval_generation.conf

# infer generation
./scripts/local/job.sh ./projects/PLATO-KAG/holle/infer_generation.conf

# multi
# calculate ppl of all candidates
./scripts/local/job.sh ./projects/PLATO-KAG/holle/eval_multi_ref_generation.conf

# eval F1 for single reference
python ./projects/PLATO-KAG/tools/generation_metrics.py \
    --refer_file $OUTPUT_PATH/test/selection/output/infer_data.tsv \
    --hypo_file ${OUTPUT_PATH}/test/generation/output/inference_output.txt

# eval ppl and F1 for multiple reference
python ./projects/PLATO-KAG/holle/eval_multi_ref_generation.py \
    --eval_output_file ${OUTPUT_PATH}/test/eval_multi/output/inference_output.txt \
    --gen_output_file ${OUTPUT_PATH}/test/generation/output/inference_output.txt \
    --infer_input_file ${OUTPUT_PATH}/test/selection/output/multi_ref_infer_data.tsv



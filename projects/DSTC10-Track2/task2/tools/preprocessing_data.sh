#!/bin/bash
set -ex

# minimal tokenize sf knowledge
python projects/DSTC10-Track2/task2/tools/preprocess_knowledge.py \
    --input_knowledge ${DATA_PATH}/knowledge.json \
    --output_knowledge ${DATA_PATH}/${DATASET_TYPE}_minimal_knowledge.json

# minimal tokenize logs
python projects/DSTC10-Track2/task2/tools/preprocess_logs.py \
    --input_logs ${DATA_PATH}/${DATASET_TYPE}_logs.json \
    --output_logs ${DATA_PATH}/${DATASET_TYPE}_minimal_logs.json

# mark locations and entities in the original logs
python projects/DSTC10-Track2/task2/tools/get_mark_logs.py \
    --locations ${DATA_PATH}/locations.json \
    --entities ${DATA_PATH}/entities.json \
    --input_logs ${DATA_PATH}/${DATASET_TYPE}_minimal_logs.json \
    --output_logs ${DATA_PATH}/${DATASET_TYPE}_marked_logs.json


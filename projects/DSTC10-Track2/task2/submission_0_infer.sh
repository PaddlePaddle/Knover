#!/bin/bash
set -e

function download_tar() {
    remote_path=$1
    local_path=$2
    if [[ ! -e $local_path ]]; then
        echo "Downloading ${local_path} ..."
        wget $remote_path

        the_tar=$(basename ${remote_path})
        the_dir=$(tar tf ${the_tar} | head -n 1)
        tar xf ${the_tar}
        rm ${the_tar}

        local_dirname=$(dirname ${local_path})
        mkdir -p ${local_dirname}

        if [[ $(readlink -f ${the_dir}) != $(readlink -f ${local_path}) ]]; then
            mv ${the_dir} ${local_path}
        fi

        echo "${local_path} has been processed."
    else
        echo "${local_path} is exist."
    fi
}

# download dataset
download_tar https://dialogue.bj.bcebos.com/Knover/projects/DSTC10-Track2/task2/data.tar data

# download models
mkdir -p models
download_tar https://dialogue.bj.bcebos.com/Knover/projects/DSTC10-Track2/task2/SOP-32L-Detection.tar models/SOP-32L-Detection
download_tar https://dialogue.bj.bcebos.com/Knover/projects/DSTC10-Track2/task2/SOP-32L-Selection.tar models/SOP-32L-Selection
download_tar https://dialogue.bj.bcebos.com/Knover/projects/DSTC10-Track2/task2/SU-32L.tar models/SU-32L

# change to the root directory of Knover
cd ../../..
export DATA_PATH="$PWD/projects/DSTC10-Track2/task2/data"
export DATASET_TYPE="test"
export MODEL_PATH="$PWD/projects/DSTC10-Track2/task2/models"
export OUTPUT_PATH="$PWD/projects/DSTC10-Track2/task2/output"

bash ./projects/DSTC10-Track2/task2/tools/preprocessing_data.sh
bash ./projects/DSTC10-Track2/task2/subtask1/infer.sh
bash ./projects/DSTC10-Track2/task2/subtask2/infer.sh
bash ./projects/DSTC10-Track2/task2/subtask3/infer.sh

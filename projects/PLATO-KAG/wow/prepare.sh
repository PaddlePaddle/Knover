#!/bin/bash
set -ex

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

# download models
# pre-trained model
# 24L NSP
# 24L SU

# fine-tuned model

# download dataset


# change to the root directory of Knover
cd ../../..
DATA_PATH="$PWD/projects/PLATO-KAG/wow/data"
MODEL_PATH="$PWD/projects/PLATO-KAG/wow/models"
OUTPUT_PATH="$PWD/projects/PLATO-KAG/wow/output"

bash ./projects/PLATO-KAG/wow/init_dual_params.sh $MODEL_PATH/24L_NSP $MODEL_PATH/24L_SU $MODEL_PATH/24L_KAG_INIT






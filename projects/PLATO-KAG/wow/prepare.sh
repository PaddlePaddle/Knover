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
mkdir -p models
# pre-trained models
download_tar https://dialogue.bj.bcebos.com/Knover/projects/PLATO-KAG/wow/24L_KAG_INIT.tar models/24L_KAG_INIT

# fine-tuned model
download_tar https://dialogue.bj.bcebos.com/Knover/projects/PLATO-KAG/wow/24L_PLATO_KAG.tar models/24L_PLATO_KAG

# download dataset
download_tar https://dialogue.bj.bcebos.com/Knover/projects/PLATO-KAG/wow/data.tar data
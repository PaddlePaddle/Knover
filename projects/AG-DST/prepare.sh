#!/bin/bash
set -eux

function download_tar() {
    remote_path=$1
    local_path=$2
    if [[ ! -e $local_path ]]; then
        echo "Downloading ${local_path} ..."
        wget --no-check-certificate $remote_path

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
download_tar https://dialogue.bj.bcebos.com/Knover/projects/AG-DST/AG-DST-24L.tar models/AG-DST-24L
# fine-tuned models
download_tar https://dialogue.bj.bcebos.com/Knover/projects/AG-DST/AG-DST-24L-multiwoz.tar models/AG-DST-24L-multiwoz
download_tar https://dialogue.bj.bcebos.com/Knover/projects/AG-DST/AG-DST-24L-woz.tar models/AG-DST-24L-woz

# download dataset
download_tar https://dialogue.bj.bcebos.com/Knover/projects/AG-DST/data.tar data

cd ../..

# preprocess
sh ./projects/AG-DST/tools/preprocess_dataset.sh
# negative sampling
sh ./projects/AG-DST/tools/negative_sampling.sh
# create sequence
sh ./projects/AG-DST/tools/create_sequence.sh

#!/bin/bash
set -eux

function download_file() {
    remote_path=$1
    local_dir=$2
    file_name=$(basename ${remote_path})
    local_path=${local_dir}/${file_name}
    echo "Downloading ${local_path} ..."
    wget --no-check-certificate ${remote_path} -P ${local_dir} -c

    if [[ ${file_name} =~ "tar" ]]; then
        tar xf ${local_path} -C ${local_dir}
        rm ${local_path}
    elif [[ ${file_name} =~ "zip" ]]; then
        unzip ${local_path} -d ${local_dir}
        rm ${local_path}
    fi

    echo "${local_path} has been processed."
}

# SMD
download_file http://nlp.stanford.edu/projects/kvret/kvret_dataset_public.zip data
download_file https://dialogue.bj.bcebos.com/Knover/projects/QKConv/QKConv_t5-large_SMD.tar models

# QReCC
download_file https://zenodo.org/record/5115890/files/qrecc-test.json data
download_file https://dialogue.bj.bcebos.com/Knover/projects/QKConv/QKConv_t5-base_QReCC.tar models

# WoW
download_file http://dl.fbaipublicfiles.com/KILT/wow-dev-kilt.jsonl data
download_file http://dl.fbaipublicfiles.com/KILT/wow-test_without_answers-kilt.jsonl data
download_file http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json models
download_file http://dl.fbaipublicfiles.com/KILT/kilt_db_simple.npz models
download_file https://dialogue.bj.bcebos.com/Knover/projects/QKConv/QKConv_bart-large_WoW.tar models

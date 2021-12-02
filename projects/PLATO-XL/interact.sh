#!/bin/bash
set -e

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
mkdir -p models
download_tar https://dialogue.bj.bcebos.com/Knover/projects/PLATO-XL/11B.tar models/11B

# change to the root directory of Knover
cd ../..

./scripts/local/job.sh ./projects/PLATO-XL/interact.conf

#!/bin/bash
################################################################################
# Run local job.
################################################################################
set -ux

if [[ $# != 1 ]]; then
    echo "usage: sh $0 job_conf"
    exit -1
fi

job_conf=$1
source ${job_conf}

# local env
export CUDA_VISIBLE_DEVICES=0,1

${job_script} ${job_conf}

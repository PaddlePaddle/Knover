#!/bin/bash
################################################################################
# Run local job.
################################################################################

if [[ $# != 1 ]]; then
    echo "usage: sh $0 job_conf"
    exit -1
fi

# local env
export CUDA_VISIBLE_DEVICES=0

job_conf=$1
source ${job_conf}

${job_script} ${job_conf}

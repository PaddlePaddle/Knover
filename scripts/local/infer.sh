#!/bin/bash
################################################################################
# Run local inference.
################################################################################

if [[ $# != 1 ]]; then
    echo "usage: sh $0 job_conf"
    exit -1
fi

# local env
export CUDA_VISIBLE_DEVICES=0,1

./scripts/distributed/infer.sh $1

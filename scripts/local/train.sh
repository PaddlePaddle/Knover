#!/bin/bash
################################################################################
# Run local training.
################################################################################

if [[ $# != 1 ]]; then
    echo "usage: sh $0 job_conf"
    exit -1
fi

# local env
export CUDA_VISIBLE_DEVICES=0,1

./scripts/distributed/train.sh $1

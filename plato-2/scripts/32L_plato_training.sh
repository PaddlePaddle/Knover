#!/bin/bash
set -eux

# change to Knover working directory
SCRIPT=`realpath "$0"`
KNOVER_DIR=`dirname ${SCRIPT}`/../..
cd $KNOVER_DIR

./scripts/local/train.sh ./package/dialog_en/plato/32L_train.conf

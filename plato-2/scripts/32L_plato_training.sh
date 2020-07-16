#!/bin/bash
set -eux

cd ..
./scripts/local/train.sh ./package/dialog_en/plato/32L_train.conf

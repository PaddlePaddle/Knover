#!/bin/bash
set -eux

cd ..
./scripts/distributed/train.sh ./package/dialog_en/plato/24L_train.conf

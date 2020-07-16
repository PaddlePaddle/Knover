#!/bin/bash
set -eux

cd ..
./scripts/local/train.sh ./package/dialog_en/plato/24L_train.conf

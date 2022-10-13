#!/bin/bash

python \
    ./knover/tools/convert_checkpoint.py \
    --convert_type paddle2pkl \
    --param_path /path/to/static-dir \
    --save_path /path/to/params.pkl

python \
    ./knover/tools/convert_checkpoint.py \
    --convert_type static2dygraph \
    --param_path /path/to/static-dir \
    --save_path /path/to/dygraph.pdparams


python \
    ./knover/tools/convert_checkpoint.py \
    --convert_type dygraph2static \
    --param_path /path/to/dygraph.pdparams \
    --save_path /path/to/static-dir

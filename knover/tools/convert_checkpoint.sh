#!/bin/bash

# paddle to pickle
python \
    ./knover/tools/convert_checkpoint.py \
    --convert_type paddle2pkl \
    --param_path /path/to/static-dir \
    --save_path /path/to/params.pkl

# paddle static to paddle dygraph
python \
    ./knover/tools/convert_checkpoint.py \
    --convert_type static2dygraph \
    --param_path /path/to/static-dir \
    --save_path /path/to/dygraph.pdparams

# paddle dygraph to paddle static
python \
    ./knover/tools/convert_checkpoint.py \
    --convert_type dygraph2static \
    --param_path /path/to/dygraph.pdparams \
    --save_path /path/to/static-dir

# paddle static fp16 to fp32
python \
    ./knover/tools/convert_checkpoint.py \
    --convert_type fp32 \
    --param_path /path/to/static-dir-fp16 \
    --save_path /path/to/static-dir-fp32

# paddle static fp32 to fp16
python \
    ./knover/tools/convert_checkpoint.py \
    --convert_type fp16 \
    --param_path /path/to/static-dir-fp32 \
    --save_path /path/to/static-dir-fp16

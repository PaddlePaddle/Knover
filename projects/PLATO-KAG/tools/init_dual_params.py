#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build initial parameters for PLATO-KAG."""

import argparse
import os
import shutil

def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--ks_param_folder", type=str, required=True)
    parser.add_argument("--rg_param_folder", type=str, required=True)
    parser.add_argument("--dst_param_folder", type=str, required=True)

    args = parser.parse_args()
    return args

def copy_file(ori_path, ori_name, new_path, new_name):
    """Copy file."""
    src = os.path.join(ori_path, ori_name)
    dst = os.path.join(new_path, new_name)
    shutil.copyfile(src, dst)

def main(args):
    """Main function."""
    ks_f = args.ks_param_folder
    rg_f = args.rg_param_folder
    dst_f = args.dst_param_folder

    ks_fs = os.listdir(ks_f)
    for f in ks_fs:
        if "moment" in f:
            continue
        if "pow" in f:
            continue
        if "post_encoder" in f:
            copy_file(ks_f, f, dst_f, "post_dual" + f[4:])
        elif f.startswith("encoder_"):
            copy_file(ks_f, f, dst_f, "dual_" + f)
        elif "embedding" in f and "mask_lm" not in f:
            copy_file(ks_f, f, dst_f, "dual_encoder_" + f)

    rg_fs = os.listdir(rg_f)
    for f in rg_fs:
        if "moment" in f:
            continue
        if "pow" in f:
            continue
        if "@LR_DECAY_COUNTER@" in f:
            continue
        if "loss_scaling" in f:
            continue
        if "num" in f:
            continue
        copy_file(rg_f, f, dst_f, f)

    print("finish dual params initialization")

if __name__ == "__main__":
    args = setup_args()
    main(args)
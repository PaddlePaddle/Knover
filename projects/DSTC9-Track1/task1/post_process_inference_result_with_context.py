#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert task1 inference output -> task1 json format output."""

import argparse
import json

from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    
    args = parser.parse_args()
    
    with open(args.pred_file) as pred_f, open(args.out_file, "w") as out_f:
        objs = []
        for line in tqdm(pred_f, desc="Generate task1 input dataset with context"):
            prob = float(line.strip().split("\t")[-1])
            objs.append({"target": prob > 0.5})
        
        json.dump(objs, out_f, indent=2)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert task3 tsv format output -> task3 json format output."""

import argparse
import json

from tqdm import tqdm


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""
    with open(args.in_file) as in_f, open(args.pred_file) as pred_f, open(args.out_file, "w") as out_f:
        preds = json.load(in_f)
        for pred in tqdm(preds, desc="Post-process task3 inference result"):
            if pred["target"]:
                candidates = []
                cols = next(pred_f).strip().split("\t")
                pred["response"] = cols[-1]
        json.dump(preds, out_f, indent=2)


if __name__ == "__main__":
    args = setup_args()
    main(args)

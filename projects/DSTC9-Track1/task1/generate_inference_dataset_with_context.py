#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert json format logs -> task1 dataset."""

import argparse
import json

from tqdm import tqdm


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--label_file", type=str, default=None)
    args = parser.parse_args()
    return args


def main(args):
    logs = json.load(open(args.log_file))
    if args.label_file is not None:
        labels = json.load(open(args.label_file))
    else:
        labels = [None for _ in logs]

    with open(args.out_file, "w") as out_f:
        if labels[0] is None:
            out_f.write("src\n")
        else:
            out_f.write("src\tlabel\n")
        for log, label in zip(tqdm(logs, desc="Generate task1 input dataset with context"), labels):
            dialog = []
            for turn in log:
                turn["text"] = turn["text"].replace("\t", " ").replace("\n", " ")
                dialog.append(f"{turn['text']}\x01{1 if turn['speaker'] == 'U' else 0}")
            dialog_str = " [SEP] ".join(dialog)
            if label is None:
                out_f.write(f"{dialog_str}\n")
            else:
                out_f.write(f"{dialog_str}\t{1 if label['target'] else 0}\n")


if __name__ == "__main__":
    args = setup_args()
    main(args)

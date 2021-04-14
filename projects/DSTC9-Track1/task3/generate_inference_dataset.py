#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert task2 json format output -> task3 tsv format input."""

import argparse
import json
import string

from tqdm import tqdm


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--knowledge_file", type=str, required=True)
    parser.add_argument("--convert_capwords", action="store_true", default=False)
    args = parser.parse_args()
    return args


def main(args):
    logs = json.load(open(args.log_file))
    knowledge = json.load(open(args.knowledge_file))

    def convert_to_knowledge(domain, entity, doc):
        field_names = ["[DOMAIN]", "[ENTITY]", "[TITLE]", "[BODY]"]
        field_values = [domain]
        if entity["name"] is not None:
            if args.convert_capwords:
                field_values.append(string.capwords(entity["name"]))
            else:
                field_values.append(entity["name"])
        else:
            field_values.append("*")
        field_values.append(" ".join(doc["title"].split()))
        field_values.append(" ".join(doc["body"].split()))
        
        fields = [name + " " + value for name, value in zip(field_names, field_values)]
        return " ".join(fields)

    with open(args.in_file) as in_f, open(args.out_file, "w") as out_f:
        out_f.write(f"src\tknowledge\n")
        preds = json.load(in_f)
        for log, pred in zip(logs, tqdm(preds, desc="Generate task3 input dataset")):
            if not pred["target"]:
                # Skip example witchout knowledge usage
                continue

            dialog = []
            for turn in log:
                utt = " ".join(turn["text"].split())
                dialog.append(f"{utt}\1{1 if turn['speaker'] == 'U' else 0}")
            dialog = " [SEP] ".join(dialog)

            k = pred["knowledge"][0]

            domain = k["domain"]
            entity_id = str(k["entity_id"])
            doc_id = str(k["doc_id"])

            entity = knowledge[domain][entity_id]
            docs = entity["docs"]
            doc = docs[doc_id]
            select_k = convert_to_knowledge(domain, entity, doc)
            out_f.write(f"{dialog}\t{select_k}\n")


if __name__ == "__main__":
    args = setup_args()
    main(args)

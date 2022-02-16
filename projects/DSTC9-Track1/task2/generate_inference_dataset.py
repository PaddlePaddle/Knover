#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert task1 json format output -> task2 tsv format input."""

import argparse
import json
import string
from tqdm import tqdm


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", required=True)
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--knowledge_file", type=str, required=True)
    parser.add_argument("--do_lowercase", action="store_true", default=False)
    parser.add_argument("--selected_domain", type=str, default=None)
    parser.add_argument("--predict_all", action="store_true", default=False)
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""
    logs = json.load(open(args.log_file))
    knowledge = json.load(open(args.knowledge_file))

    def __convert_to_tgt(domain, entity, doc):
        field_names = ["[DOMAIN]", "[ENTITY]", "[TITLE]", "[BODY]"]
        field_values = [domain]
        if entity["name"] is not None:
            field_values.append(entity["name"])
        else:
            field_values.append("*")
        field_values.append(" ".join(doc["title"].split()))
        field_values.append(" ".join(doc["body"].split()))
        if args.do_lowercase:
            field_values = [x.lower() for x in field_values]
        
        fields = [name + " " + value for name, value in zip(field_names, field_values)]
        return " ".join(fields)

    with open(args.in_file) as in_f, open(args.out_file, "w") as out_f:
        out_f.write("src\ttgt\n")
        preds = json.load(in_f)
        for log, pred in zip(logs, tqdm(preds, desc="Generate task2 input dataset")):
            if not args.predict_all and not pred["target"]:
                # Skip example witchout knowledge usage.
                continue

            if args.selected_domain is not None and pred["knowledge"][0]["domain"] != args.selected_domain:
                # Only select one domain.
                continue

            dialog = []
            for turn in log:
                turn["text"] = " ".join(turn["text"].split())
                if args.do_lowercase:
                    turn["text"] = turn["text"].lower()
                dialog.append(f"{turn['text']}\x01{1 if turn['speaker'] == 'U' else 0}")
            dialog = " [SEP] ".join(dialog)

            for domain in knowledge:
                for entity_id in knowledge[domain]:
                    entity = knowledge[domain][entity_id]
                    for doc_id in entity["docs"]:
                        doc = entity["docs"][doc_id]
                        tgt = __convert_to_tgt(domain, entity, doc)
                        out_f.write(f"{dialog}\t{tgt}\1{0}\n")


if __name__ == "__main__":
    args = setup_args()
    main(args)

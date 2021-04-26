#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert json format logs -> task1 dataset."""

import argparse
import json
import string

from tqdm import tqdm


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--knowledge_file", type=str, required=True)
    parser.add_argument("--schema_desc_file", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""
    with open(args.log_file) as f:
        logs = json.load(f)
    with open(args.knowledge_file) as f:
        knowledge = json.load(f)
    with open(args.schema_desc_file) as f:
        schema_desc = json.load(f)

    def __convert_to_tgt(domain, entity, doc):
        field_names = ["[DOMAIN]", "[ENTITY]", "[TITLE]", "[BODY]"]
        field_values = [domain]
        if entity["name"] is not None:
            field_values.append(entity["name"])
        else:
            field_values.append("*")
        field_values.append(" ".join(doc["title"].split()))
        field_values.append(" ".join(doc["body"].split()))

        field_values = [x.lower() for x in field_values]
        fields = [name + " " + value for name, value in zip(field_names, field_values)]
        return " ".join(fields)

    def __convert_schema_desc_to_tgt(doamin, knowledge_type, description):
        field_names = ["[DOMAIN]", "[TYPE]", "[DESC]"]
        field_values = [domain, knowledge_type, description]

        field_values = [x.lower() for x in field_values]
        fields = [name + " " + value for name, value in zip(field_names, field_values)]
        return " ".join(fields)

    with open(args.out_file, "w") as out_f:
        out_f.write("src\ttgt\n")
        for log in tqdm(logs, desc="Generate task1 input dataset with schema"):
            dialog = []
            for turn in log:
                utt = " ".join(turn["text"].split())
                role_id = 1 if turn['speaker'] == 'U' else 0
                dialog.append(f"{utt.lower()}\1{role_id}")
            dialog_str = " [SEP] ".join(dialog)

            for domain in knowledge:
                for entity_id in knowledge[domain]:
                    entity = knowledge[domain][entity_id]
                    for doc_id in entity["docs"]:
                        doc = entity["docs"][doc_id]
                        tgt = __convert_to_tgt(domain, entity, doc)
                        out_f.write(f"{dialog_str}\t{tgt}\1{0}\n")

            for domain in schema_desc:
                for desc in schema_desc[domain]:
                    knowledge_type = schema_desc[domain][desc]

                    tgt = __convert_schema_desc_to_tgt(domain, knowledge_type, desc)
                    out_f.write(f"{dialog_str}\t{tgt}\1{0}\n")


if __name__ == "__main__":
    args = setup_args()
    main(args)

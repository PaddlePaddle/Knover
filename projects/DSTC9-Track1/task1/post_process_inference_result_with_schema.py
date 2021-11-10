#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert task1 inference output -> task1 json format output."""

import argparse
import json

from tqdm import tqdm


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--log_file", type=str, required=True)
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

    with open(args.pred_file) as pred_f, open(args.out_file, "w") as out_f:
        preds = []
        for i, log in enumerate(tqdm(logs[:5], desc="Post-process task1 inference result")):
            candidates = []
            for domain in knowledge:
                for entity_id in knowledge[domain]:
                    x = knowledge[domain][entity_id]
                    for doc_id in x["docs"]:
                        line = next(pred_f).strip().split("\t")
                        prob = float(line[-1])
                        candidates.append((prob, "QA", domain, entity_id, doc_id))

            for domain in schema_desc:
                for description in schema_desc[domain]:
                    knowledge_type = schema_desc[domain][description]
                    line = next(pred_f).strip().split("\t")
                    prob = float(line[-1])
                    if knowledge_type == "service":
                        continue
                    candidates.append((prob, "TASK", domain, knowledge_type, description))
            candidates = sorted(candidates, key=lambda x: -x[0])

            if candidates[0][1] == "QA":
                candidates = list(filter(lambda x: x[1] == "QA", candidates))[:5]
                pred = {
                    "target": True,
                    "knowledge": [
                        {
                            "domain": domain,
                            "entity_id": entity_id if entity_id == "*" else int(entity_id),
                            "doc_id": int(doc_id),
                            "prob": prob
                        }
                        for prob, _, domain, entity_id, doc_id in candidates
                    ]
                }
            else:
                pred = {"target": False}
            preds.append(pred)
        json.dump(preds, out_f, indent=2)


if __name__ == "__main__":
    args = setup_args()
    main(args)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert task2 tsv format output -> task2 json format input."""

import argparse
import json

from tqdm import tqdm


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--knowledge_file", type=str, required=True)
    parser.add_argument("--predict_all", action="store_true", default=False)
    args = parser.parse_args()
    return args


def main(args):
    knowledge = json.load(open(args.knowledge_file))

    with open(args.in_file) as in_f, open(args.pred_file) as pred_f, open(args.out_file, "w") as out_f:
        preds = json.load(in_f)
        for pred in tqdm(preds, desc="Post-process task2 inference result"):
            if not args.predict_all and not pred["target"]:
                continue

            candidates = []
            for domain in knowledge:
                for entity_id in knowledge[domain]:
                    x = knowledge[domain][entity_id]
                    for doc_id in x["docs"]:
                        line = next(pred_f).strip().split("\t")
                        prob = float(line[-1])
                        candidates.append((prob, domain, entity_id, doc_id))

            if pred["target"]:
                pred["knowledge"] = [
                    {
                        "domain": domain,
                        "entity_id": entity_id if entity_id == "*" else int(entity_id),
                        "doc_id": int(doc_id),
                        "prob": prob
                    }
                    for prob, domain, entity_id, doc_id in sorted(candidates, reverse=True)[:5]
                ]
        json.dump(preds, out_f, indent=2)


if __name__ == "__main__":
    args = setup_args()
    main(args)

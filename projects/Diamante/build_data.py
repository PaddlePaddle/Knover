from collections import Counter
import random
import json


def get_F1(string, sub):
    """calculate F1 score"""

    common = Counter(string) & Counter(sub)
    overlap = sum(common.values())
    recall, precision = overlap / len(sub), overlap / len(string)
    return (2 * recall * precision) / (recall + precision + 1e-12)


def write_train_tsv(out_path, paddle_data):
    """write data"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("src\ttgt\tlabel\n")
        for line in paddle_data:
            line = [l.replace("\n", " ").replace("\t", " ") for l in line]
            assert len(line) == 3
            f.write("\t".join(line) + "\n")


def build_data(file_name, epoches=1):
    """build training data"""

    neg_pool = []
    for dialog in open(file_name, encoding="utf-8"):
        dialog = json.loads(dialog)
        neg_pool.extend([conv["utterance"] for conv in dialog["conversation"]])

    data = []
    for epoch in range(epoches):

        epoch_data = []
        for dialog in open(file_name, encoding="utf-8"):

            dialog = json.loads(dialog)
            context = [conv["utterance"] for conv in dialog["conversation"]]
            for index in range(1, len(context)):

                src = context[:index]
                src = " [SEP] ".join(src)

                human_reply = context[index]
                random_reply = neg_pool[random.randint(0, len(neg_pool) - 1)]

                bot_reply = [b for b in dialog["conversation"][index]["response_candidates"]
                             if b not in ["", human_reply]]
                similarity = [[get_F1(human_reply, b), b] for b in bot_reply]
                bot_reply = sorted(similarity, key=lambda x: x[0], reverse=False)
                bot_reply = [b[1] for b in bot_reply[:5]]
                random.shuffle(bot_reply)

                if bot_reply == []:
                    continue
                bot_reply = bot_reply[0]

                pairs = []
                # human / bot / random
                pairs.append([src, human_reply, "1"])
                pairs.append([src, bot_reply, "0"])
                pairs.append([src, random_reply, "0"])

                epoch_data.append(pairs)
     
        if "train" in file_name:
            random.shuffle(epoch_data)

        data.extend(epoch_data)

    ob = []
    for pairs in data:
        ob.extend(pairs)

    return ob


data = build_data("./projects/Diamante/luge_Diamante/train.txt", epoches=5)
write_train_tsv("./projects/Diamante/processed_data/train.tsv", data)

data = build_data("./projects/Diamante/luge_Diamante/valid.txt", epoches=1)
write_train_tsv("./projects/Diamante/processed_data/valid.tsv", data)

data = build_data("./projects/Diamante/luge_Diamante/test.txt", epoches=1)
write_train_tsv("./projects/Diamante/processed_data/test.tsv", data)

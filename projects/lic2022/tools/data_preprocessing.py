#encoding=utf-8
"""
LIC2022 DuSinc dataset preprocessing
"""

import sys
import json

def conv_to_gen_query(fin_file, fout_file, is_test=False):
    """
    原始数据集转换为Query生成模型训练所需的格式
    """
    fout = open(fout_file, "w", encoding="utf-8")
    if is_test:
        fout.write("src\n")
    else:
        fout.write("src\ttgt\n")
    with open(fin_file, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            context = []
            topical = " ".join(data["user_topical"])
            location = data["user_location"]
            for uttr in data["conversation"]:
                utterence = uttr["utterance"]
                if is_test:
                    context.append(utterence)
                    continue
                if uttr["role"] == "bot":
                    if "use_query" in uttr:
                        query = uttr["use_query"]
                    else:
                        query = "不 检索"
                    outstr = location + " [SEP] " + " [SEP] ".join(context) + "\t" + query
                    fout.write(outstr.strip().replace("\n", " ") + "\n")
                context.append(utterence)
            if is_test:
                outstr = location + " [SEP] " + " [SEP] ".join(context)
                fout.write(outstr.strip().replace("\n", " ") + "\n")
    fout.close()

def conv_to_gen_response(fin_file, fout_file, is_test=False):
    """
    原始数据集转换为知识对话生成模型训练所需的格式
    """
    fout = open(fout_file, "w", encoding="utf-8")
    if is_test:
        fout.write("knowledge\tsrc\n")
    else:
        fout.write("knowledge\tsrc\ttgt\n")
    with open(fin_file, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            context = []
            topical = " ".join(data["user_topical"])
            location = data["user_location"]
            if is_test:
                context = [uttr["utterance"] for uttr in data["conversation"][:-1]]
                if "use_knowledge" in data["conversation"][-1]:
                    knowledge = data["conversation"][-1]["use_knowledge"]
                else:
                    knowledge = ""
                # 对部分过长的知识进行截断，只保留前256个字符
                knowledge = knowledge.replace("\n", " ").replace("\t", " ")[:256]
                outstr = knowledge + "\t" + location + " [SEP] " + " [SEP] ".join(context)
                fout.write(outstr.rstrip().replace("\n", " ") + "\n")
                continue
            for uttr in data["conversation"]:
                if is_test:
                    context.append(uttr["utterance"])
                    continue
                if "use_kg_label" in uttr:
                    if uttr["use_kg_label"] == "true":
                        try:
                            knowledge = uttr["use_knowledge"].replace("\n", " ").replace("\t", " ")
                        except:
                            print(json.dumps(uttr, ensure_ascii=False, indent=2))
                    else:
                        knowledge = ""
                    response = uttr["utterance"]
                    outstr = knowledge + "\t" + location + " [SEP] " + " [SEP] ".join(context) + "\t" + response
                    fout.write(outstr.rstrip().replace("\n", " ") + "\n")    
                context.append(uttr["utterance"])
    fout.close()

conv_to_gen_query("DuSinc_release/train.txt", "preprocess_data/train_query.txt")
conv_to_gen_response("DuSinc_release/train.txt", "preprocess_data/train_dial.txt")
conv_to_gen_query("DuSinc_release/dev.txt", "preprocess_data/dev_query.txt")
conv_to_gen_response("DuSinc_release/dev.txt", "preprocess_data/dev_dial.txt")
conv_to_gen_query("DuSinc_release/test_query_1.txt", "preprocess_data/test_query_1.txt", is_test=True)
conv_to_gen_response("DuSinc_release/test_dial_1.txt", "preprocess_data/test_dial_1.txt", is_test=True)

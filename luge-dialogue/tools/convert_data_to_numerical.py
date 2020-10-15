#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import re
import sys
import json
import random
import sentencepiece as spm

sys.stdin.reconfigure(encoding="utf8")
sys.stdout.reconfigure(encoding="utf8")

if len(sys.argv) < 2:
    print("Usage: " + sys.argv[0] + " sentence_piece_model")
    exit()

sp = spm.SentencePieceProcessor()
sp.load(sys.argv[1])

type_dict = {"chitchat": 30001, "knowledge": 30002, "recommend": 30003}

#0: both(context and respone) meet the requirements
#1: last utterance of context is too long and trunc
#2: first utterance of context is too long and trunc
#3: other utterance of context is too long and trunc
#4: respone is too long  and trunc
truncate_type_stat = {0:0, 1:0, 2:0, 3:0, 4:0}

def truncate_ids_list(ids_list, cut_len=512, truncate_first_turn=False):
    if sum([len(x) for x in ids_list]) <= cut_len:
        return 0, ids_list
    
    new_ids_list = []
    ids_list.reverse()
    len_cnt = 0
    cut_type = 0

    for  i, ids in enumerate(ids_list):
        if len_cnt + len(ids) > cut_len:
            if len_cnt == 0 and (len(ids_list) > 1 or not truncate_first_turn):
                new_ids_list.append(ids[-cut_len:])
                len_cnt = cut_len
                cut_type = 1 # last utterance of context is too long
            elif truncate_first_turn and i == len(ids_list) - 1 and len_cnt + 1 < cut_len:
                new_ids_list.append(ids[:cut_len - len_cnt - 1] + [ids[-1]])
                len_cnt = cut_len
                cut_type = 2 # first utterance of context is too long and trunc
            else:
                cut_type = 3 # other utterance of context is too long and trunc
            break
        else:
            len_cnt += len(ids)
            new_ids_list.append(ids)

    new_ids_list.reverse()
    return cut_type, new_ids_list

def convert_sample_to_numerical(input_data, max_seq_len=512, max_response_len=128, truncate_first_turn=False, is_test=False):
    assert "type" in input_data and "context" in input_data and "response" in input_data and "knowledge" in input_data

    for key in sample:
        sample[key] = re.sub("  +", " ", sample[key])

    data_type = input_data["type"]
    context = input_data["context"]
    response = input_data["response"]
    knowledge = input_data["knowledge"]

    # type
    assert data_type in type_dict
    type_id = type_dict[data_type]

    # tokenize response
    response_ids = sp.EncodeAsIds(response) + [2]
    if len(response_ids) > max_response_len - 1:
        new_response_ids = response_ids[1 - max_response_len:]
        truncate_type_stat[4] += 1
        if not is_test:
            return None
    else:
        new_response_ids = response_ids[:]

    # tokenize context
    context_ids_list = []
    if knowledge != "":
        knowledge_ids = sp.EncodeAsIds(knowledge) + [2]
        context_ids_list.append(knowledge_ids)
    
    if context != "":
        for utterance in context.split('\t'):
            utterance_ids = sp.EncodeAsIds(utterance) + [2]
            context_ids_list.append(utterance_ids)
    
    truncate_type, new_context_ids_list = truncate_ids_list(context_ids_list, max_seq_len - max_response_len - 2, truncate_first_turn=truncate_first_turn)
    truncate_type_stat[truncate_type] += 1

    if truncate_type == 1 and not is_test:
        return None

    # deal context tokens
    token_ids = [1, type_id]
    sent_ids = [0, 0]
    for ids in new_context_ids_list:
        token_ids += (ids)
        sent_ids += ([0] * len(ids))
    
    # deal reply tokens
    token_ids += [1]
    sent_ids += [1]
    if not is_test:
        token_ids += new_response_ids
        sent_ids += ([1] * len(new_response_ids))
    
    assert(len(token_ids) == len(sent_ids))
    position_ids = range(len(token_ids))

    output_list = []
    for l in [token_ids, sent_ids, position_ids]:
        output_list.append(' '.join([str(x) for x in l]))
    
    return output_list

def to_sample_for_douban(input_file, data_type="chitchat", is_test=False):
    with open(input_file, encoding='utf8') as fp:
        for line in fp:
            data = json.loads(line.strip())

            history = data["history"]
            response = data["response"] if "response" in data else ""

            if not is_test:
                sample = {"type": data_type,
                          "knowledge": "",
                          "context": history,
                          "response": response}

                yield sample
            else:
                sample = {"type": data_type,
                          "knowledge": "",
                          "context": '\t'.join(history),
                          "response": response}

                yield sample

def to_sample_for_lccc(input_file, data_type="chitchat", is_test=False):
    with open(input_file, encoding='utf8') as fp:
        for line in fp:
            data = json.loads(line.strip())
            if not is_test:
                conversation = data["conversation"]
                
                for i in range(1, len(conversation)):
                    sample = {"type": data_type,
                              "knowledge": "",
                              "context": '\t'.join(conversation[:i]),
                              "response": conversation[i]}
                    
                    yield sample
            else:
                history = data["history"]
                response = data["response"] if "response" in data else ""

                sample = {"type": data_type,
                          "knowledge": "",
                          "context": '\t'.join(history),
                          "response": response}

                yield sample
    
def to_sample_for_weibo(input_file, data_type="chitchat", is_test=False):
    with open(input_file, encoding='utf8') as fp:
        for line in fp:
            data = json.loads(line.strip())

            history = data["history"]
            response = data["response"] if "response" in data else ""

            if not is_test:
                sample = {"type": data_type,
                        "knowledge": "",
                        "context": history,
                        "response": response}

                yield sample

            else:
                sample = {"type": data_type,
                        "knowledge": "",
                        "context": '\t'.join(history),
                        "response": response}

                yield sample
    
def to_sample_for_duconv(input_file, data_type="knowledge", is_test=False):
    with open(input_file, encoding='utf8') as fp:
        for line in fp:
            data = json.loads(line.strip())

            goal = data["goal"]
            knowledge = data["knowledge"]

            goal_knowledge = ' '.join([' '.join(spo) for spo in goal + knowledge])

            if not is_test:
                conversation = data["conversation"]

                for i in range(0, len(conversation), 2):
                    sample = {"type": data_type, 
                              "knowledge": goal_knowledge,
                              "context": '\t'.join(conversation[:i]) if i > 0 else "",
                              "response": conversation[i]}

                    yield sample
            else:
                history = data["history"]
                response = data["response"] if "response" in data else ""

                sample = {"type": data_type,
                          "knowledge": goal_knowledge,
                          "context": '\t'.join(history),
                          "response": response}

                yield sample
    
def to_sample_for_kdconv(input_file, data_type="knowledge", is_test=False):
    with open(input_file, encoding='utf8') as fp:
        for line in fp:
            data = json.loads(line.strip())
            knowledge = data["knowledge"]

            knowledge = ' '.join([' '.join(spo) for spo in knowledge])

            if not is_test:
                conversation = data["conversation"]
                for i in range(len(conversation)):
                    sample = {"type": data_type,
                              "knowledge": knowledge,
                              "context": '\t'.join(conversation[:i]) if i > 0 else "",
                              "response": conversation[i]}

                    yield sample
            else:
                history = data["history"]
                response = data["response"] if "response" in data else ""

                sample = {"type": data_type,
                          "knowledge": knowledge,
                          "context": '\t'.join(history),
                          "response": response}

                yield sample
    
def to_sample_for_tencent(input_file, data_type="knowledge", is_test=False):
    with open(input_file, encoding='utf8', errors="ignore") as fp:
        for line in fp:
            data = json.loads(line.strip())

            knowledge = data["knowledge"]
            history = data["history"]
            response = data["response"] if "response" in data else ""
            
            if not is_test:
                knowledge = ' '.join(knowledge)
                sample = {"type": data_type,
                          "knowledge": knowledge,
                          "context": history,
                          "response": response}

                yield sample

            else:
                knowledge = ' '.join([' '.join(item) for item in knowledge])
                sample = {"type": data_type,
                          "knowledge": knowledge,
                          "context": '\t'.join(history),
                          "response": response}

                yield sample
    
def to_sample_for_durecdial(input_file, data_type="recommend", is_test=False):
    def goal_processing(goal):
        format_goal = []
        while isinstance(goal, list):
            goal = goal[0]
        goal = goal.split('-->')
        for i, g in enumerate(goal):
            format_g = []
            g = g.strip()
            si, ei = g.find('['), g.find(']')
            if si != 0 or ei <= si+1 or not g[si+1:ei].isdigit():
                continue
            
            g = g.split(g[si:ei+1])[-1]
            g_n = g.split('(', 1)[0].strip()
            g_d = g.split('(', 1)[-1].strip()

            format_g.append(g_n)

            if "新闻" in g_n or g_n.replace(' ', '') in ["关于明星的聊天", "兴趣点推荐", "音乐推荐", "播放音乐", "美食推荐", "poi推荐", "电影推荐", "音乐点播", "问日期", "新闻推荐", "新闻点播", "问答"]:
                left = -1
                for right, c in enumerate(g_d):
                    if c == "『":
                        left = right + 1
                    elif c == "』":
                        if left >= 0 and right > left:
                            item = g_d[left:right].strip()
                            if item not in format_g and item.replace(' ', '') != "参考知识":
                                format_g.append(item)
                        left = -1
            
            format_goal.append(format_g)
        
        if len(format_goal) > 3:
            format_goal = [format_goal[0], format_goal[-2], format_goal[-1]]

        return format_goal

    def user_profile_processing(user_profile):
        accept_key = ["拒绝", "喜欢的电影", "喜欢的明星", "喜欢的poi", "喜欢的音乐", "喜欢的新闻", "同意的新闻", "同意的音乐", "同意的美食", "同意的poi", "同意的电影"]
        format_user_profile = []
        for key in user_profile:
            if key.replace(' ', '') in accept_key:
                if isinstance(user_profile[key], list):
                    format_user_profile.append([key, ' '.join(user_profile[key])])
                else:
                    format_user_profile.append([key, user_profile[key]])
        
        return format_user_profile

    def strip_utterance(utterance_list):
        for i, utterance in enumerate(utterance_list):
            utterance = utterance.split(' ')
            if re.match("\[\d+\]", utterance[0]) is not None:
                utterance = utterance[1:]
            utterance = ' '.join(utterance)
            utterance_list[i] = utterance

    with open(input_file, encoding='utf8') as fp:
        for line in fp:
            data = json.loads(line.strip())

            situation = data["situation"]
            goal = data["goal"]
            user_profile = data["user_profile"]
            knowledge = data["knowledge"]

            goal = goal_processing(goal)
            user_profile = user_profile_processing(user_profile)

            goal = ' '.join([' '.join(g) for g in goal])
            user_profile = ' '.join([' '.join(up) for up in user_profile])
            knowledge = ' '.join([' '.join(spo) for spo in knowledge])

            background = ' '.join([goal, situation, user_profile, knowledge])

            if not is_test:
                conversation = data["conversation"]
                
                bot_mode = 0 if goal[0][0] == '寒暄' else 1
                
                strip_utterance(conversation)

                for i, utterance in enumerate(conversation):
                    if i % 2 != bot_mode:
                        continue
                    
                    sample = {"type": data_type,
                              "knowledge": background,
                              "context": '\t'.join(conversation[:i]) if i > 0 else "",
                              "response": conversation[i]}

                    yield sample
            else:
                history = data["history"]
                response = data["response"] if "response" in data else ""

                strip_utterance(history)

                sample = {"type": data_type,
                          "knowledge": background,
                          "context": '\t'.join(history),
                          "response": response}

                yield sample

if __name__ == '__main__':
    # change the input and output files to your real files
    data_process_list = [
                            [
                                [
                                    ["./data/luge-dialogue/weibo/train.txt", to_sample_for_weibo, False, False],
                                    ["./data/luge-dialogue/douban/train.txt", to_sample_for_douban, False, False],
                                    ["./data/luge-dialogue/LCCC/LCCD_train.json", to_sample_for_lccc, False, False],
                                    ["./data/luge-dialogue/duconv/train.txt", to_sample_for_duconv, True, False],
                                    ["./data/luge-dialogue/kdconv/train.txt", to_sample_for_kdconv, True, False],
                                    ["./data/luge-dialogue/tencent/train.txt", to_sample_for_tencent, True, False],
                                    ["./data/luge-dialogue/DuRecDial/train.txt", to_sample_for_durecdial, True, False]
                                ],
                                "./data/train.txt",
                            ],
                            [
                                [
                                    ["./data/luge-dialogue/weibo/dev.txt", to_sample_for_weibo, False, False],
                                    ["./data/luge-dialogue/douban/dev.txt", to_sample_for_douban, False, False],
                                    ["./data/luge-dialogue/LCCC/LCCD_dev.json", to_sample_for_lccc, False, False],
                                    ["./data/luge-dialogue/duconv/dev.txt", to_sample_for_duconv, True, False],
                                    ["./data/luge-dialogue/kdconv/dev.txt", to_sample_for_kdconv, True, False],
                                    ["./data/luge-dialogue/tencent/dev.txt", to_sample_for_tencent, True, False],
                                    ["./data/luge-dialogue/DuRecDial/dev.txt", to_sample_for_durecdial, True, False]
                                ],
                                "./data/valid.txt",
                            ],
                            [
                                [
                                    ["./data/luge-dialogue/weibo/test.txt", to_sample_for_weibo, False, True],
                                    ["./data/luge-dialogue/douban/test.txt", to_sample_for_douban, False, True],
                                    ["./data/luge-dialogue/LCCC/test.txt", to_sample_for_lccc, False, True],
                                    ["./data/luge-dialogue/duconv/test.txt", to_sample_for_duconv, True, True],
                                    ["./data/luge-dialogue/kdconv/test.txt", to_sample_for_kdconv, True, True],
                                    ["./data/luge-dialogue/tencent/test.txt", to_sample_for_tencent, True, True],
                                    ["./data/luge-dialogue/DuRecDial/test.txt", to_sample_for_durecdial, True, True]
                                ],
                                "./data/test.txt",
                            ],
                        ]
    for [input_list, output_file] in data_process_list:
        truncate_type_stat[0] = truncate_type_stat[1] = truncate_type_stat[2] = truncate_type_stat[3] = truncate_type_stat[4] = 0
        fout = open(output_file, 'w')
        for [input_file, handle_method, truncate_first_turn, is_test] in input_list:
            for sample in handle_method(input_file, is_test=is_test):
                numerical = convert_sample_to_numerical(sample, truncate_first_turn=truncate_first_turn, is_test=is_test)
                if numerical is not None:
                    fout.write(';'.join(numerical) + "\n")   
        fout.close()

        T = truncate_type_stat[0] + truncate_type_stat[1] + truncate_type_stat[2] + truncate_type_stat[3] + truncate_type_stat[4]
        FT = float(T)
        T1 = truncate_type_stat[1]
        T2 = truncate_type_stat[2]
        T3 = truncate_type_stat[3]
        T4 = truncate_type_stat[4]
        sys.stderr.write('Total num : %d \n\ttruncate type 1: %d rate(%.4f)\n\ttruncate tye 2: %d rate(%.4f)\n\ttruncate type 3: %d rate(%.4f)\n\ttruncate type 4: %d rate(%.4f)\n' % (T, T1, (T1/FT), T2, (T2/FT), T3, (T3/FT), T4, (T4/FT)))


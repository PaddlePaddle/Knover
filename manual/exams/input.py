from collections import namedtuple

fields = ["token_ids", "type_ids", "pos_ids", "tgt_start_idx", "data_id"]
Record = namedtuple("Record", fields, defaults=(None,) * len(fields))
# 新的会话
question = '我刚刚去动物园了。'
# 历史对话
history = '你好？[SEP] 你好，谈谈你自己吧？[SEP] 我今天过得很开心呢！[SEP] 是嘛，你今天干了什么？'
# 背景知识
background = '天气：晴朗，地点：北京，人物：男孩，动物园，熊猫'
# 回答
answer = '北京动物园吧，那里的熊猫很受欢迎呢！'

question = SPM.encode_as_ids(question)
history = [SPM.encode_as_ids(text) for text in history.split("[SEP]")]
background = SPM.encode_as_ids(background)
answer = SPM.encode_as_ids(answer)

token_ids = []
type_ids = []
data_id = 0  # 如果样本多的话，会按排序进行标记

token_ids += [0] + background + [2]  # 0 表示语句开头[BOS]，2表示句子结尾[EOS]
type_ids += [0] + len(background) * [0] + [0]  # 0 表示context类， 1表示response类

for line in history:
    token_ids += line + [2]
    type_ids += len(line) * [0] + [0]

token_ids += question + [2]
type_ids += len(question) * [0] + [0]

token_ids += [0] + answer + [2]
type_ids += [1] + len(answer) * [1] + [1]  # 注意符号的变化

fields_value = {}
fields_value["token_ids"] = token_ids
fields_value["type_ids"] = type_ids
fields_value["pos_ids"] = list(range(len(type_ids)))
fields_value["tgt_start_idx"] = fields_value["pos_ids"][-1]
fields_value["data_id"] = data_id

record = Record(**fields_value)
print(record)

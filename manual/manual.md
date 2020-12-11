# 这是一个比较详尽的Knover使用手册
该项目是从项目集“没有对象就自己造”提炼总结出来的，感兴趣的请参考[链接](https://aistudio.baidu.com/aistudio/projectdetail/542430)的“没有对象就自己造板块”。这个集合主要的目的是记录我关于Knover的使用心得，并为以后使用这个工具的人提供一些快速入门的建议。其中难免有一些错误的地方，还请见谅。
# 1 什么是Knover
Knover是基于飞桨的开放域对话模型训练工具箱。通过Knover，我们可以轻松训练出一个开放域对话模型，并支持多种输入格式，支持单文件和多文件读取。同时，Knover提供了基于飞桨的优秀的优化训练策略，加速模型的训练。目前，Knover已经支持PLATO-2模型的训练。
Knover官网：https://github.com/PaddlePaddle/Knover 。
# 2 什么是PLATO-2
Plato是百度推出的一个基于飞桨的多轮对话模型。该模型的最大改进在于通过引入离散隐变量实现了对话场景中多回答中的择优，即，对于同一个问题，实现不同场景下的不同回答的选择。最新推出的Plato-2在中英文效果上，已全面超越 Google Meena、Facebook Blender、微软小冰等先进模型。
模型的整体框架如图所示。该模型采用了经典的Transformer结构，通过注意力机制提高了模型针对不同长度对话的生成效果。隐变量z的引入，使预训练模型依据z可以生成多种回答，最终回答从多种回答中择优。在训练中，该模型采用两阶段训练的方法。第一阶段，基于表现良好的预训练模型，训练出一对一回答的模型；第二阶段，引入评估和隐变量z，训练出一对多回答的模型。模型的具体原理可以参考原论文，论文地址：https://arxiv.org/abs/2006.16779 。
![Plato-2](https://github.com/fiyen/PaddlePaddle-Knover/blob/main/pictures/Plato-2ModelReview.png)
# 3 认识Knover
## 3.1 认识Knover的主要文件
对于Knover的使用，用命令行是比较方便的（当然如果有能力可以调用sh文件）。根目录的三个文件：train.py,test.py和save_inference_model.py是最经常用到的。从名称也可以看出来，train.py用来训练，test.py用来测试，save_inference_model用来导出预测用的模型，这一个是在模型训练完以后，部署训练好的模型，又希望保持模型尽可能缩减不必要的参数时用的。

1.package文件夹中存放了其自带的试验数据的词集，语句切分模型（spm.model, 即sentencepiece model，这个模型用在语句的预处理上，必须给定），以及模型的细节参数（词集大小，隐含层个数，激活函数等等，具体查看package/dialog_en/24L.json。
2.models文件夹存放了模型的各个子模块，plato模块也在其中
3.data文件夹存放了实验用的小文件
4.tasks文件夹中包含了模型两种应用任务的实现，包括“下一句语句预测”和“会话生成”。这个应用任务的选择是必须给出的，对应参数 `--tasks`, 分别写作`NextSentencePrediction`和`DialogGeneration`。具体来说，`DialogGeneration`是训练对话模型时用到的，而`NextSentencePrediction`是在训练打分模型时用到的。这里的具体区别后边再讲。

## 3.2 认识Konver的主要参数
`--init_pretraining_params` 预训练模型所在文件夹，如果需要加载（当然需要）预训练模型需要给出这个参数

`--init_checkpoint` 保存节点的所在文件夹，如果给出了init_checkpoint则从该文件夹初始化训练参数（等于覆盖了init_pretraining_params的参数），checkpoint保存了模型更多的细节，如训练步数，当前学习率，还有模型涉及的所有训练参数等，如果从checkpoint开始继续训练模型，模型会从之前中断的状态继续训练，如果不设`--start_step`模型会错误显示当前的步数，但是内部的参数是按照正确的步数更新的。

`--max_seq_len`: 最长输入长度

`--is_distributed`: 是否进行分布式训练，即用多显卡进行加速


train.py

`--train_file` 训练文件地址

`--valid_file` 评估文件地址

`--model` 用到的模型名称：`Plato`：plato；`NSPModel`：next_sentence_prediction model；`UnifiedTransformer`

`--config_path` 模型细节参数配置文件，如24L.json

`--task` 模型应用任务 `NextSentencePrediction`；`DialogGeneration`

`--vocab_path` 词集路径

`--spm_model_file` sentencepiece model文件的路径

`--num_epochs` 训练周期数

`--log_steps` 输出训练详情的间隔步数

`--validation_steps` 评价间隔步数

`--save_steps` 保存间隔步数

`--start_step`: 训练开始步长，用于中断后继续训练，不设置也不会影响训练，但是会造成输出的步数是错的

`--batch_size`: 训练中的批数据大小

`--lr` `--warmup_steps` `weight_decay`: 学习率相关选项

infer.py

`--infer_file` 需要推断的文件

`--output_name` 需要保存的对象，`response`；`data_id`；`score`

`--model` 用到的模型名称：`Plato`：plato；`NSPModel`：next_sentence_prediction model；`UnifiedTransformer`

`--config_path` 模型细节参数配置文件，如24L.json

`--task` 模型应用任务 `NextSentencePrediction`；`DialogGeneration`；
`--vocab_path` 词集路径

`--spm_model_file` sentencepiece model文件的路径

## 3.3 了解对话模型的训练
### 3.3.1 一般模型的训练
第一步：在进行训练之前，需要提前准备好几个文件：详列模型参数的.json文件，如config文件夹下的24L.json文件；分词工具的预训练模型，如spm.model文件；以及分词后形成的词表文件，如vocab.txt。

第二步：准备数据。把自己准备的训练数据转换成合适的格式，存入txt文件，供训练使用。

第三步：调用train.py进行训练，模型选择`UnifiedTransformer`，任务选择`DialogGeneration`。

第四步：训练完成后，调用save_inference_model.py将预测模型导出

经过以上四步，模型就训练好了。当然这个过程需要巨量的训练集做支撑才能训练出好的模型。
### 3.3.2 Plato-2模型的训练
在上述四步完成后进行。前两步与上述过程相似，在训练出UnifiedTransformer模型后，按照以下步骤进行训练：

第三步：调用train.py进行训练，模型选择`Plato`，任务选择`DialogGeneration`，并且`--init_pretraining_params`选择之前训练好的UnfiedTransformer模型（如果未进行上述第四步，则导出的模型可以用`--init_checkpoint`指定）

第四步：训练完成后，继续调用train.py进行训练，模型选择`Plato`，任务选择`NextSentencePrediction`（注意区别）。训练打分模型。

第五步：分别用save_inference_model.py导出PLATO模型和打分模型NSP。

经过这些步，模型训练完成。用infer.py预测时，如果使用PLATO模型，需要指定打分方式`--ranking_score`，如果选择`nsp_score`，则需要设定打分模型为NSP。以上具体过程后边会细讲。

# 4 具体操作
## 4.1 数据准备
Plato-2模型的输入采用了token，role，turn，position相融合的表示方式。在训练和测试过程中，我们需要搞清楚文本数据需要经过怎样的转化才能作为输入，以及输出数据需要怎样的处理才能转换成文本。目前我们可以获取各种开放的对话训练集，如微博，腾讯，华为等提供的比赛用的数据集。
### 4.1.1 中文分词
中文必须面对的一个问题就是如何实现分词。在公开的开放域对话数据集中，大多数已经做了分词，然而真实场景中语句是不可能时时刻刻都被分词了的。在Knover的源码中，对输入的处理是通过了sentencepiece工具（BERT也使用的这个）。sentencepiece提供了一种方便快捷的分词操作，我们可以直接将整个数据集放进去，设定分词的单元数量，然后等待训练出一个好用的分词模型（会输出一个预训练模型，之后对每个语句都会用这个模型进行编码和解码，即分词，转换成数字编码，输出再转换回句子）。Knover中训练必须输入的参数spm_model，就是对应sentencepiece的预训练模型。我们当然可以自己训练一个sentencepiece的预训练模型出来，但是考虑到分词模型对效果的影响，推荐大家使用千言多技能对话中给出的baseline模型（luge-dialogue）中附带的spm.model文件，这个文件分词的效果已经非常出色了。当然，别忘了搭配词表vocab.txt使用。目前这个比赛已经关闭，luge-dialogue这个模块可以在Konver官网获得。

仔细分析luge的spm.model我们可以发现，这个预训练模型其实是根据已经分词的句子训练的，虽说如此，因为分词单元足够多，也覆盖了所有常见的单个中文词。我们可以直接把语句送入得到编码，也可以先用jieba分词预先分一次（也可以用其他分词工具），然后再编码。用sentencepiece模型的例子如下（文件exams/do_sentencepiece.py）：

```
import sentencepiece as sp
import jieba
text = "我今天做了一顿丰盛的晚餐！"

SPM = sp.SentencePieceProcessor()
SPM.load('spm.model')
# 直接分词
ids = SPM.encode_as_ids(text)
print(ids)
print(SPM.decode_ids(ids))

# 先用jieba分词，再用sentencepiece编码
text = ' '.join(list(jieba.cut(text)))
ids = SPM.encode_as_ids(text)
print(ids)
print(SPM.decode_ids(ids))
```
### 4.1.2 文本的输入
Plato对文本输入的支持还是挺多样化的，它支持直接输入原始文本，也支持输入经过tokenize的分词序列，或者是已经编码（encoded）的数字序列。但是无论Plato支持的格式如何，在进行训练和预测之前，都会转换成能够被识别的标准格式。在Knover中，这个格式是通过定义的Record完成的。Record的定义如下：
```
from collections import namedtuple
Record = namedtuple("Record", fields, defaults=(None,) * len(fields))
```
在解释fields的值之前，我们先来思考一下Plato需要哪些输入。在完成一段对话时，我们通常会综合对话的历史和自己所知的历史知识来进行判断，来决定自己将要回答什么。而在对话生成中，这些信息也是需要考虑的。因此，Plato需要的输入有两个，首先，是当前对方的问话，其次是已经进行过的历史对话信息，最后是背景知识。由于各种条件的限制，背景知识可能并没有办法获取，所以至少需要的是已进行的历史对话信息，和此时对方的问话。进一步，我们需要考虑更多的信息：如果纪录了历史对话，我们如何判断每段对话的起始位置，如何判断从什么时候开始生成需要的回答，在训练集中，我们还要知道哪一部分是训练中给出的回答用于调整模型的参数。
![PlatoInput](https://github.com/fiyen/PaddlePaddle-Knover/blob/main/pictures/PLATOInput.png)
上图给出了Plato模型需要的输入，当然这些是以Embedding的形式给出的，而Embedding是在模型中转化的，它在转化之前是以数字编码存在的。Embedding现在已经是语言处理技术的标配了，它把每一个标记映射到空间中，增加其表征能力。我们暂时忽略最前边的latent，它是表示不同回答方式的因变量，用于Plato在众多可能回答中选择正确的回答，我们这里不关心这个是怎么实现的，所以不展开讨论。在latent之后，有contex和response两个内容，其中context包含了众多信息：历史对话，背景知识，以及对话与对话之间分隔的符号[EOS/EOU], [BOS/BOS]等等，如果有背景知识的话，也会列到context中。response则是训练中需要的部分，在预测中这一部分是空的。

TokenEmbeddings表示各语言单元的Embedding（词向量）；RoleEmbeddings是各个语言单元在其中扮演的角色，这个主要是用来区分内容是context（EA）还是response（EB）（亦或是背景知识，背景知识可以作为response的角色，也可以单独成为一类，即EC）；TurnEmbeddings表示每一部分在当前回合中的相对回合数（PLATO2中已经不存在这一项）；PositionEmbeddings则是每个语言单元的位置，一般是range(0, len(context))。

知道了这些，我们回到Record上来看这个输入应该怎么得到。由定义可知，Record是带名称的元组，这样我们立马可以知道，这个元组是通过名称来调用其中的内容的。fields的内容是什么呢？从官方的源码中可以总结出：fields = ["token_ids", "type_ids", "pos_ids", "tgt_start_idx", "data_id"]。也就是说，输入需要给出5个部分，token_ids就是处理过的语言单位的编码；type_ids就是个语言单位扮演的角色，是context还是response；pos_ids是各个语言单位的位置；tgt_start_idx是回复生成的开始位置，也即context的最后一个词的位置；data_id就是这个训练样本的标记。
如下给出一个例子，可以清楚的知道一个输入是如何形成的（文件exams/input.py）：
```
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
```
### 4.1.3 Knover支持的输入
`--data_format`: 输入的文件内容形式

`raw`: 未进行处理标记和编码的纯文本（.txt, .tsv）每行代表一个样本，对话之间用"[SEP]"分开，对于训练样本，需要用"\t"隔开并加上回复语句。

例子："你是谁？[SEP] 你猜！[SEP] 别闹！  我是你的小甜甜！" 

或者分词后的："你 是 谁 ？[SEP] 你 猜 ！[SEP] 别 闹 ！ 我 是 你的 小甜甜 ！"

'tolenized': 进行标记的文本（.txt, .tsv）每行代表一个样本，对话之间用"[SEP]"分开，对于训练样本，需要用"\t"隔开并加上回复语句。

例子："你_ 是_ 谁_ ？[SEP] 你_ 猜_ ！[SEP] 别_ 闹_ ！  我_ 是_ 你的 小甜甜 !"，

需要注意的是，标记后的形式是如何的与选择的vocab.txt有关，具体要看vocab.txt内标记的词是什么样子的。

'numerical': 进行编码的数字。必须输入的有三个类型（token_ids, type_ids, pos_ids）并依次用";"隔开，token_ids需要标记句子的开始（1）和结束（0）。而是否有回复则是用type_ids来标记的。

例子：

训练样本表示为"1 23 45 3000 45 89 0 11 32 33 89 0; 0 0 0 0 0 0 0 1 1 1 1 1; 0 1 2 3 4 5 6 7 8 9 10 11"；

预测用样本为"1 23 45 3000 45 89 0; 0 0 0 0 0 0 0 1 ; 0 1 2 3 4 5 6 7"，注意在type_ids中加"1"用以定位tgt_start_ids。

`--file_format`: 输入的文件组织形式

`file`: 只有一个文件。

`filelist`: 多个文件，所有文件都记录在形如train_filelist的文件里，表明多个文件的地址，每一行为一个文件。
## 4.2 定义配置
由于在训练模型的时候，需要输入--config_path，这个参数用来读取模型的配置（Transformer层数量等等），这里我们需要定义两个模型的配置文件（**.json）。如下参数生成两个配置文件，配置即为我数据集中附带的模型的配置，如果有兴趣和算力，可以自己改配置训练，最有效的参数是num_hidden_layers和num_attention_heads，增加这些值会增加模型的规模。
```
import json

## 定义UnifiedTransformer的参数
key = {'pre_encoder_cmd': 'd', 'preprocess_cmd': 'n', 'postprocess_cmd': 'da', 'post_cls_cmd': 'n', 'cls_bias': True,
 'attention_probs_dropout_prob': 0.1, 'hidden_act': 'gelu', 'hidden_dropout_prob': 0.1, 'hidden_size': 768, 
 'initializer_range': 0.02, 'max_position_embeddings': 512, 'num_attention_heads': 12, 
 'num_hidden_layers': 12, 'type_vocab_size': 2, 'role_type_size': 32, 'vocab_size': 30004}
f = open("12L.json", "w")
json_data = json.dump(key, f)
f.close()

## 定义Plato的参数
key = {'pre_encoder_cmd': 'd', 'preprocess_cmd': 'n', 'postprocess_cmd': 'da', 'post_cls_cmd': 'n', 'cls_bias': True,
 'attention_probs_dropout_prob': 0.1, 'hidden_act': 'gelu', 'hidden_dropout_prob': 0.1, 'hidden_size': 768, 
 'initializer_range': 0.02, 'max_position_embeddings': 512, 'latent_type_size': 20, 'num_attention_heads': 12, 
 'num_hidden_layers': 12, 'type_vocab_size': 2, 'role_type_size': 32, 'vocab_size': 30004}
f = open("12L_P.json", "w")
json_data = json.dump(key, f)
f.close()
```

建议可操作参数：

`hidden_size`: 隐含层尺寸。

`max_position_embedding`: 最大位置编码，规定了可接收样本的最大长度，也即输入token_ids的最大长度，太长会被截断。

`num_attention_heads`: 多头注意力的数量，参考注意力机制。

`num_hidden_layers`: Transformer层数，参考注意力机制。

`vocab_size`: 词表规模，即有多少个可编码的词单元。

`latent_type_size`: 仅Plato模型可设置，即隐变量z的规模，决定了文本生成阶段生成回答的次数，生成后用打分机制选取最好的回答。

## 4.3 训练命令
训练UnifiedTransformer：
```
python Knover/train.py \
--model UnifiedTransformer --task DialogGeneration --vocab_path Knover/config/vocab.txt --spm_model_file Knover/config/spm.model \
--train_file pro_data/train.txt --valid_file pro_data/valid.txt --data_format numerical --file_format file --config_path Knover/config/12L.json \
--init_checkpoint Knover/latest_model/ut_model \
--in_tokens True --batch_size 16000 -lr 1e-5 --warmup_steps 1000 --weight_decay 0.01 --num_epochs 20 \
--max_src_len 384 --max_tgt_len 128 --max_seq_len 512 \
--log_step 100 --validation_steps 20000 --save_steps 5000 \
--save_path Knover/output \
--is_distributed False \
--is_cn True \
--start_step ??
```

训练PLATO:
```
python Knover/train.py \
--model Plato --task DialogGeneration --vocab_path Knover/config/vocab.txt --spm_model_file Knover/config/spm.model \
--train_file pro_data/train.txt --valid_file pro_data/valid.txt --data_format numerical --file_format file --config_path Knover/config/12L_P.json \
--init_checkpoint Knover/latest_model/pt_model \
--in_tokens True --batch_size 1000 -lr 1e-5 --warmup_steps 1000 --weight_decay 0.01 --num_epochs 10 \
--max_src_len 384 --max_tgt_len 128 --max_seq_len 512 \
--log_step 100 --validation_steps 20000 --save_steps 100 \
--save_path Knover/output \
--is_cn True
```

训练NSPModel打分模型:
```
python Knover/train.py \
--model NSPModel --task NextSentencePrediction --vocab_path Knover/config/vocab.txt --spm_model_file Knover/config/spm.model \
--train_file pro_data/train.txt --valid_file pro_data/valid.txt --data_format numerical --file_format file --config_path Knover/config/12L_P.json \
--init_checkpoint Knover/latest_model/pt_model \
--in_tokens True --batch_size 1000 -lr 4*1e-4 --warmup_steps 1000 --weight_decay 0.01 --num_epochs 10 \
--max_src_len 384 --max_tgt_len 128 --max_seq_len 512 \
--log_step 100 --validation_steps 20000 --save_steps 100 \
--save_path Knover/output \
--mix_negative_sample True
```
如果希望提速，可以在百度AiStudio上用脚本进行，代码如下（文件exams/distributed_training.py，也可以直接访问https://aistudio.baidu.com/aistudio/clusterprojectdetail/1154630 进行fork后运行）
```
# coding=utf-8

###### 欢迎使用脚本任务,让我们首选熟悉下一些使用规则吧 ###### 

# 数据集文件目录
datasets_prefix = '/root/paddlejob/workspace/train_data/datasets/'

# 数据集文件具体路径请在编辑项目状态下,通过左侧导航栏「数据集」中文件路径拷贝按钮获取
train_datasets =  '通过路径拷贝获取真实数据集文件路径 '

# 输出文件目录. 任务完成后平台会自动把该目录所有文件压缩为tar.gz包，用户可以通过「下载输出」可以将输出信息下载到本地.
output_dir = "/root/paddlejob/workspace/output"

# 日志记录. 任务会自动记录环境初始化日志、任务执行日志、错误日志、执行脚本中所有标准输出和标准出错流(例如print()),用户可以在「提交」任务后,通过「查看日志」追踪日志信息

import os

if __name__ == '__main__':
    
    print(os.getcwd())
    print("预装依赖包")
    os.system("pip install -i https://mirror.baidu.com/pypi/simple --upgrade pip")
    os.system("pip install -i https://mirror.baidu.com/pypi/simple sentencepiece")
    print("解压Knover模块")
    #os.system("unzip /root/paddlejob/workspace/train_data/datasets/data56424/Knover.zip")
    os.system("unzip /root/paddlejob/workspace/train_data/datasets/data57647/Knover.zip")
    os.system("unzip /root/paddlejob/workspace/train_data/datasets/data57647/model.zip")
    os.system("unzip /root/paddlejob/workspace/train_data/datasets/data57647/NSP.zip")
    os.system("unzip /root/paddlejob/workspace/train_data/datasets/data57647/test_2.zip")
    print("解压数据集")
    #os.system("unzip /root/paddlejob/workspace/train_data/datasets/data56424/pro_data.zip")
    
    print("开始训练")
    os.system("export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7")
    os.system("cd ./home/aistudio/Knover/latest_model/ && ls -a")
    
    #####################################################PARAMETERS######################################################
    epochs = 1
    start_step = 0
    lr = 1e-4
    model = "Plato"
    batch_size = 1
    model_name = "pt_model"  # {"ut_model": UnifiedTransformer, "pt_model": Plato}
    in_tokens = True  # infer.py 中设为False
    
    config_name = "12L_P.json"  # {"12L.json", "12L_P.json", "24L.json", "24L_P.json"}, P is for Plato model
    
    func_py = "infer.py"  # {"train.py", "infer.py"}
    
    split_size = 5000  # infer.py 运行时切分文件所含样本最大数量
    ####################################################################################################################
    
    if func_py == 'train.py':
        if model == 'Plato' or model == 'UnifiedTransformer':
            args = "--model {} --task DialogGeneration --vocab_path ./home/aistudio/Knover/config/vocab.txt --spm_model_file ./home/aistudio/Knover/config/spm.model \
                --train_file ./home/aistudio/pro_data/train.txt --valid_file ./home/aistudio/pro_data/valid.txt --data_format numerical --file_format file --config_path ./home/aistudio/Knover/config/{} \
                --in_tokens {} --batch_size {} -lr {} --warmup_steps 1000 --weight_decay 0.01 --num_epochs {} \
                --max_src_len 384 --max_tgt_len 128 --max_seq_len 512 \
                --log_step 100 --validation_steps 5000 --save_steps 100 \
                --is_distributed True is_cn True --start_step {} \
                --init_checkpoint ./model/{} \
                --save_path /root/paddlejob/workspace/output \
                ".format(model, config_name, in_tokens, batch_size, lr, epochs, start_step, model_name)
            os.system("python -m paddle.distributed.launch ./home/aistudio/Knover/{} {}".format(func_py, args))
        elif model == 'NSPModel':
            args = "--model {} --task NextSentencePrediction --vocab_path ./home/aistudio/Knover/config/vocab.txt --spm_model_file ./home/aistudio/Knover/config/spm.model \
                --train_file ./home/aistudio/pro_data/train.txt --valid_file ./home/aistudio/pro_data/valid.txt --data_format numerical --file_format file --config_path ./home/aistudio/Knover/config/{} \
                --in_tokens {} --batch_size {} -lr {} --warmup_steps 1000 --weight_decay 0.01 --num_epochs {} \
                --max_src_len 384 --max_tgt_len 128 --max_seq_len 512 \
                --log_step 100 --validation_steps 5000 --save_steps 100 \
                --is_distributed True --start_step {} \
                --init_checkpoint ./model/{} \
                --save_path /root/paddlejob/workspace/output \
                --mix_negative_sample True \
                ".format(model, config_name, in_tokens, batch_size, lr, epochs, start_step, model_name)
            os.system("python -m paddle.distributed.launch ./home/aistudio/Knover/{} {}".format(func_py, args))
        else:
            raise ValueError("Only support Plato, UnifiedTransformer, and NSPModel but received %s" % model)
```
## 4.4 导出模型
保存UnifiedTransformer
```
python Knover/save_inference_model.py \
--model UnifiedTransformer \
--do_generation true \
--task DialogGeneration \
--vocab_path Knover/config/vocab.txt --spm_model_file Knover/config/spm.model \
--init_checkpoint Knover/latest_model/pt_model \
--inference_model_path UnifiedTransformerModel \
--config_path Knover/config/12L.json
```

保存NSPModel:
```
python Knover/save_inference_model.py \
--model NSPModel \
--task NextSentencePrediction \
--vocab_path Knover/config/vocab.txt --spm_model_file Knover/config/spm.model \
--init_checkpoint Knover/latest_model/nsp_model \
--inference_model_path NSP \
--config_path Knover/config/12L_P.json
```

保存Plato
```
python Knover/save_inference_model.py \
--model Plato \
--do_generation true \
--task DialogGeneration \
--vocab_path Knover/config/vocab.txt --spm_model_file Knover/config/spm.model \
--init_checkpoint Knover/latest_model/pt_model \
--inference_model_path Plato \
--config_path Knover/config/12L_P.json
```

## 4.5 预测
```
python Knover/infer.py \
--model Plato --task DialogGeneration --vocab_path Knover/config/vocab.txt --spm_model_file Knover/config/spm.model \
--infer_file pro_data/test.txt --data_format numerical --file_format file --config_path Knover/config/12L_P.json \
--init_checkpoint Knover/latest_model/pt_model \
--batch_size 1 \
--max_src_len 384 --max_tgt_len 128 --max_seq_len 512 \
--output_name response \
--do_generation True --num_samples 20 --topk 5 --is_cn True \
--save_path Knover/output --log_step 100
```
当然，也可以通过时生成的NSPModel和Plato来预测，代码如下：
```
python Knover/infer.py \
--model Plato --task DialogGeneration --vocab_path Knover/config/vocab.txt --spm_model_file Knover/config/spm.model \
--infer_file pro_data/test.txt --data_format numerical --file_format file --config_path Knover/config/12L_P.json \
--init_pretraining_params Plato --nsp_inference_model_path NSP --ranking_score nsp_score \
--batch_size 1 \
--max_src_len 384 --max_tgt_len 128 --max_seq_len 512 \
--output_name response \
--do_generation True --num_samples 20 --topk 5 --is_cn True \
--do_generation true --save_path Knover/output --log_step 1
```

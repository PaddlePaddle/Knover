import sentencepiece as sp
import jieba
text = "我今天做了一顿丰盛的晚餐！"

# 先设置spm.model的路径
spm_path = 'spm.model'
SPM = sp.SentencePieceProcessor()
SPM.load(spm_path)
# 直接分词
ids = SPM.encode_as_ids(text)
print(ids)
print(SPM.decode_ids(ids))

# 先用jieba分词，再用sentencepiece编码
text = ' '.join(list(jieba.cut(text)))
ids = SPM.encode_as_ids(text)
print(ids)
print(SPM.decode_ids(ids))

import os
from enum import Enum
from utils.config import Configer

# 全局配置

config = Configer()

proj_path = os.path.dirname(__file__)

# bert_path = os.path.join(proj_path, 'data/emb/bert-mini/')
emb_path = proj_path
#emb_path = "/home/wujinjie/kesci_question_multilabel_classification"
bert_path = os.path.join(emb_path, "data/emb/chinese_L-12_H-768_A-12/")
# 语料相关配置
fold_data_path = os.path.join(proj_path, "data/textcnn/data/fold_data.pl")

# emb

char_emb_path = os.path.join(proj_path, "data/wiki_100.utf8")
word_emb_path = os.path.join(proj_path, "data/emb/industry_vec.txt")

baike_vocab_path = os.path.join(proj_path, "data/BaikeWordList.txt")
baike_vec_path = os.path.join(proj_path, "data/w2v.h5")

stop_word_path = os.path.join(proj_path, "data/stopwords.txt")

# 从第几个epoch开始resume训练，如果为0，从头开始
RESUME_EPOCH = 0

# from submodules.emo_classifier import *
# from submodules.ner_classifier import *
# from submodules.gc_transformer import *
from submodules.tf_bert import *
import numpy as np
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath
from gensim import corpora
from transformers import BertTokenizer
import pickle
import os

mTokenizer = BertTokenizer.from_pretrained("klue/bert-base")
mGC_tokenizer = pickle.load(open(os.environ['CHATBOT_ROOT'] + "/resources/converters/tokenizer.pickle", 'rb'))
mNER_tokenizer = pickle.load(open(os.environ['CHATBOT_ROOT'] + "/resources/converters/letter_to_index.pickle", 'rb'))
VOCAB_SIZE = mGC_tokenizer.vocab_size + 2

emotion_labels = {"불만": 0, "중립": 1, "당혹": 2, "기쁨": 3, "걱정": 4, "질투": 5, "슬픔": 6, "죄책감": 7, "연민": 8}
theme_labels = {"날씨 및 계절": 0, "여행": 1, "가족": 2, "스포츠_레저": 3, "게임": 4, "건강": 5, "무주제": 6, "영화_만화" : 7, "식음료": 8,
                "학교": 9, "반려동물": 10, "교통": 11, "연예": 12, "무주제": 13}

emotion_mapping_by_index = dict((value, key) for (key, value) in emotion_labels.items())
theme_mapping_by_index = dict((value, key) for (key, value) in theme_labels.items())
index_mapping_by_NER = {'O': 0, 'B-LC': 1, 'I-LC': 2, 'B-QT': 3, 'I-QT': 4, 'B-OG': 5, 'I-OG': 6, 'B-DT': 7, 'I-DT': 8, 
                        'B-PS': 9, 'I-PS': 10, 'B-TI': 11, 'I-TI': 12}
NER_mapping_by_index = {0 : '0', 1 : 'B-LC', 2 : 'I-LC', 3 : 'B-QT', 4 : 'I-QT', 5 : 'B-OG', 6 : 'I-OG', 7 : 'B-DT', 
                        8 : 'I-DT', 9 : 'B-PS', 10 : 'I-PS', 11 : 'B-TI', 12 : 'I-TI', 13 : 'UNK'}
NER_labels = dict((value, key) for (key, value) in NER_mapping_by_index.items())

from emo_classifier import *
from ner_classifier import *
from tf_bert import *
from gc_transformer import *
import numpy as np
from transformers import BertTokenizer
import pickle
import os

mTokenizer = BertTokenizer.from_pretrained("klue/bert-base")
mGC_tokenizer = pickle.load(open(os.environ['CHATBOT_ROOT'] + "/resources/converters/tokenizer.pickle", 'rb'))
VOCAB_SIZE = mGC_tokenizer.vocab_size + 2

emotion_labels = {"불만": 0, "중립": 1, "당혹": 2, "기쁨": 3, "걱정": 4, "질투": 5, "슬픔": 6, "죄책감": 7, "연민": 8}

emotion_mapping_by_index = dict((value, key) for (key, value) in emotion_labels.items())
NER_mapping_by_index = {0: 'O', 1: 'PER-B', 2: 'PER-I', 3: 'FLD-B', 4: 'FLD-I', 5: 'AFW-B', 6: 'AFW-I', 7: 'ORG-B',
                        8: 'ORG-I', 9: 'LOC-B',
                        10: 'LOC-I', 11: 'CVL-B', 12: 'CVL-I', 13: 'DAT-B', 14: 'DAT-I', 15: 'TIM-B', 16: 'TIM-I',
                        17: 'NUM-B',
                        18: 'NUM-I', 19: 'EVT-B', 20: 'EVT-I', 21: 'ANM-B', 22: 'ANM-I', 23: 'PLT-B', 24: 'PLT-I',
                        25: 'MAT-B',
                        26: 'MAT-I', 27: 'TRM-B', 28: 'TRM-I'}
NER_labels = dict((value, key) for (key, value) in NER_mapping_by_index.items())

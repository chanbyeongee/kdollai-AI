from AIKoBERT import get_tokenizer, convert_examples_to_features, ner_prediction
import pandas as pd
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pickle
from transformers import TFBertForTokenClassification

labels = [label.strip() for label in open('C:\MyProjects\Other Projects\AI ChatBot Project/ner_label.txt', 'r', encoding='utf-8')]

tag_size = len(labels)

print(tag_size)

tokenizer = get_tokenizer()

new_model = TFBertForTokenClassification.from_pretrained("klue/bert-base", num_labels=tag_size+1, from_pt=True)

new_model.load_weights("C:\MyProjects\Other Projects\AI ChatBot Project\Training_Model_Weights/NER_KoBERT_Keras/NER_KoBERT")

with open('C:\MyProjects\Other Projects\AI ChatBot Project\Training_Model_Weights/NER_RNN/index_to_tag.pickle', 'rb') as f:
  index_to_tag = pickle.load(f)

test_samples = ['5원으로 맺어진 애인까지 돈이라는 민감한 원자재를 통해 현대인의 물질만능주의를 꼬집고 있는 이 무비는 .']
result_list = ner_prediction(test_samples, tokenizer, new_model, index_to_tag, max_seq_len=128)
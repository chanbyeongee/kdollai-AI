import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import f1_score, classification_report
from tensorflow.keras.utils import to_categorical

from AIKoBERT import TFBertForTokenClassification, get_tokenizer, convert_examples_to_features

train_data = pd.read_csv("C:\MyProjects\Other Projects\AI ChatBot Project/ner_train_data.csv")
test_data = pd.read_csv("C:\MyProjects\Other Projects\AI ChatBot Project/ner_test_data.csv")
# print(train_data)

print('챗봇 샘플의 개수 :', len(train_data))

# NaN이 있는지 check
print(train_data.isnull().sum())
train_data = train_data.dropna(how = 'any')
print(train_data.isnull().sum())
print(test_data.isnull().sum())
test_data = test_data.dropna(how = 'any')
print(test_data.isnull().sum())

# 입력과 라벨을 각각 띄어쓰기 기준으로 split시켜 list 타입으로 저장
train_data_sentence = [sent.split() for sent in train_data['Sentence'].values]
test_data_sentence = [sent.split() for sent in test_data['Sentence'].values]
train_data_label = [tag.split() for tag in train_data['Tag'].values]
test_data_label = [tag.split() for tag in test_data['Tag'].values]

# 29개의 개체명 라벨들 load
labels = [label.strip() for label in open('C:\MyProjects\Other Projects\AI ChatBot Project/ner_label.txt', 'r', encoding='utf-8')]
print('개체명 태깅 정보 :', labels)

# 개체명 라벨과 정수를 상호 변환할 수 있도록 두 dictionary 정의
tag_to_index = {tag: index for index, tag in enumerate(labels)}
index_to_tag = {index: tag for index, tag in enumerate(labels)}

# tag_size(나중에 num layer 개수) 정의
tag_size = len(tag_to_index)
print('개체명 태깅 정보의 개수 :',tag_size)

# tokenizer load
tokenizer = get_tokenizer()

# 데이터셋 생성
X_train, y_train = convert_examples_to_features(train_data_sentence, train_data_label, max_seq_len=128, tokenizer=tokenizer, vocab=tag_to_index)
X_test, y_test = convert_examples_to_features(test_data_sentence, test_data_label, max_seq_len=128, tokenizer=tokenizer, vocab=tag_to_index)

# 원핫인코딩
one_hot_y_train = to_categorical(y_train)

# 모델 생성 및 컴파일
model = TFBertForTokenClassification("klue/bert-base", labels=tag_size+1)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.CategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# 훈련
model.fit(
    X_train, one_hot_y_train, epochs=2, batch_size=32, validation_split=0.1
)
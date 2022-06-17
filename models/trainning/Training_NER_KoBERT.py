import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import f1_score, classification_report
from tensorflow.keras.utils import to_categorical

from AIKoBERT import TFBertForTokenClassification, convert_examples_to_features, NER_make_datasets
from KHUDoll_AIModels import get_converters

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

# tokenizer load
converters = get_converters()
tokenizer = converters[0]
# 29개의 개체명 라벨들 load
labels = converters[4]

print('개체명 태깅 정보 :', labels)

# 개체명 라벨과 정수를 상호 변환할 수 있도록 두 dictionary 정의
tag_to_index = {tag: index for index, tag in enumerate(labels)}
index_to_tag = {index: tag for index, tag in enumerate(labels)}

# tag_size(나중에 num layer 개수) 정의
tag_size = len(tag_to_index)
print('개체명 태깅 정보의 개수 :',tag_size)

# 데이터셋 생성
X_train, y_train = NER_make_datasets(train_data['Sentence'], train_data_label, max_len=128, tokenizer=tokenizer, converter=tag_to_index)
X_test, y_test = NER_make_datasets(test_data['Sentence'], test_data_label, max_len=128, tokenizer=tokenizer, converter=tag_to_index)

# 최대 길이: 128
input_id = X_train[0][0]
attention_mask = X_train[1][0]
token_type_id = X_train[2][0]
label = y_train[0]

print('단어에 대한 정수 인코딩 :',input_id)
print('어텐션 마스크 :',attention_mask)
print('세그먼트 인코딩 :',token_type_id)
print('각 인코딩의 길이 :', len(input_id))
print('정수 인코딩 복원 :',tokenizer.decode(input_id))
print('레이블 :',label)

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
import pandas as pd
import numpy as np
import os
from transformers import shape_list, BertTokenizer, TFBertModel
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
from AIKoBERT import TFBertForSequenceClassification, label_to_index, EMO_make_datasets
from KHUDoll_AIModels import get_converters

train_ner_df = pd.read_csv("/content/감성대화말뭉치_최종_전반.csv", engine='python',encoding='CP949')

X_train = train_ner_df['문장']
y_train = train_ner_df['감정']

y_train_encoded = y_train

# 라벨을 정수로 변환
i = 0
for label in y_train:
  index = label_to_index(label)
  y_train_encoded[i] = index
  i += 1

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train_encoded, test_size=0.1, random_state=0)

print(X_train)
print(y_train)

# tokenizer load
converters = get_converters()
tokenizer = converters[0]

max_len = 128

train_X, train_y = EMO_make_datasets(X_train, y_train, max_len=max_len, tokenizer=tokenizer)
test_X, test_y = EMO_make_datasets(X_test, y_test, max_len=max_len, tokenizer=tokenizer)

model = TFBertForSequenceClassification("klue/bert-base", num_labels=9)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])

history = model.fit(
    train_X, train_y, epochs=2, batch_size=64, validation_split=0.1
)
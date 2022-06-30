from tensorflow.keras.utils import to_categorical
from bertmodels import  AIModel
import pandas as pd
import tensorflow as tf
import os

train_data = pd.read_csv(os.environ['CHATBOT_ROOT']+"/resources/training_data/ner_train_data.csv")
test_data = pd.read_csv(os.environ['CHATBOT_ROOT']+"/resources/training_data/ner_test_data.csv")
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

#위에는 데이터 불러오기

main_model = AIModel()

# 데이터셋 생성
X_train, y_train = main_model.NER_make_datasets(train_data['Sentence'], train_data_label)
X_test, y_test = main_model.NER_make_datasets(test_data['Sentence'], test_data_label)

# 최대 길이: 128
input_id = X_train[0][0]
attention_mask = X_train[1][0]
token_type_id = X_train[2][0]
label = y_train[0]

print('단어에 대한 정수 인코딩 :',input_id)
print('어텐션 마스크 :',attention_mask)
print('세그먼트 인코딩 :',token_type_id)
print('각 인코딩의 길이 :', len(input_id))
print('레이블 :',label)

# 원핫인코딩
one_hot_y_train = to_categorical(y_train)


#훈련시키기
# 모델 생성 및 컴파일

model = TokenClassification("klue/bert-base", labels=len(AIModel.NER_labels)+1)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.CategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# 훈련
model.fit(
    X_train, one_hot_y_train, epochs=2, batch_size=32, validation_split=0.1
)


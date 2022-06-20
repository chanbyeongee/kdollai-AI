import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 훈련 데이터셋 구조 생성 함수(훈련을 위함)
#다대다
def NER_make_datasets(sentences, labels, max_len, tokenizer, converter):

  input_ids, attention_masks, token_type_ids, labels_list = [], [], [], []

  for sentence, label in zip(sentences, labels):
    # 문장별로 정수 인코딩 진행
    input_id = tokenizer.encode(sentence, max_length=max_len)
    # encode한 정수들의 수만큼 1로 할당
    attention_mask = [1] * len(input_id)
    # 입력(문장)이 1개이므로 세그먼트 임베딩의 모든 차원이 0
    token_type_id = [0] * max_len
    # label을 정수로 convert
    indexs = []
    for one_word, one_label in zip(sentence.split(), label):
      # label그대로 정답데이터를 만드는 것 보다, 한 단어들 모두 subword로 나뉘어서 인코딩 되므로
      # 원래 단어 위치에 맞게 label index를 넣어주고, subword로 생긴 자리에는 상관 없는 수(29)를 할당해주면서 정답데이터를 만든게 정답률이 높음
      sub_words = tokenizer.tokenize(one_word)

      ############### KHUDOLL AIMODEL 75Lines tokenizer word dictionary
      indexs.extend([converter[one_label]] + [29] * (len(sub_words) - 1))

    indexs = indexs[:max_len]

    input_ids.append(input_id)
    attention_masks.append(attention_mask)
    token_type_ids.append(token_type_id)
    labels_list.append(indexs)

  # 패딩
  input_ids = pad_sequences(input_ids, padding='post', maxlen=max_len)
  attention_masks = pad_sequences(attention_masks, padding='post', maxlen=max_len)
  labels_list = pad_sequences(labels_list, padding='post', maxlen=max_len, value=29)

  input_ids = np.array(input_ids, dtype=int)
  attention_masks = np.array(attention_masks, dtype=int)
  token_type_ids = np.array(token_type_ids, dtype=int)
  labels_list = np.asarray(labels_list, dtype=np.int32)

    #텐서, 어텐션, 세그먼트, 답  => Training_NER_KoBERT 50번줄
  return (input_ids, attention_masks, token_type_ids), labels_list


# 장사
# 예측하려는 입력 문장을 BERT 입력 구조로 변환하는 함수
# 다대일(감정)
#예측을 위한 함수(실제 상황에서 실시간으로 들어가는 함수)
#구별하는이유는 정답데이터의 유무
def NER_make_datasets_for_prediction(sentences, max_len, tokenizer):

  input_ids, attention_masks, token_type_ids, index_positions = [], [], [], []

  for sentence in sentences:
    # 문장별로 정수 인코딩 진행
    input_id = tokenizer.encode(sentence, max_length=max_len)
    # encode한 정수들의 수만큼 1로 할당
    attention_mask = [1] * len(input_id)
    # 입력(문장)이 1개이므로 세그먼트 임베딩의 모든 차원이 0
    token_type_id = [0] * max_len
    # label을 정수로 convert
    indexs = []
    for one_word in sentence.split():
      # 하나의 단어가 시작되는 지점을 1, subword로 생긴 자리나, pad된 부분을 29으로 표시한다. 이는 예측된 label의 자리를 나타낸 것이다.
      sub_words = tokenizer.tokenize(one_word)
      indexs.extend([1] + [29] * (len(sub_words) - 1))

    indexs = indexs[:max_len]

    input_ids.append(input_id)
    attention_masks.append(attention_mask)
    token_type_ids.append(token_type_id)
    index_positions.append(indexs)

  # 패딩
  input_ids = pad_sequences(input_ids, padding='post', maxlen=max_len)
  attention_masks = pad_sequences(attention_masks, padding='post', maxlen=max_len)
  index_positions = pad_sequences(index_positions, padding='post', maxlen=max_len, value=29)

  input_ids = np.array(input_ids, dtype=int)
  attention_masks = np.array(attention_masks, dtype=int)
  token_type_ids = np.array(token_type_ids, dtype=int)
  index_positions = np.asarray(index_positions, dtype=np.int32)

# index_positions은 1이면 정답이 있을곳 29면 아님
  return (input_ids, attention_masks, token_type_ids), index_positions


#실제 예측하는 함수
def ner_predict(inputs, tokenizer, model, converter, max_len=128):
  # 입력 데이터 생성
  input_datas, index_positions = NER_make_datasets_for_prediction(inputs, max_len=max_len, tokenizer=tokenizer)
  # 예측
  raw_outputs = model.predict(input_datas)
  # 128 x 29 차원의 원핫 인코딩 형태로 확률 예측값이 나오므로 최댓값만을 뽑아내 128차원 벡터로 변환
  outputs = np.argmax(raw_outputs, axis = -1)
  #### 감정에도 적용가능할듯

  pred_list = []
  result_list = []

  for i in range(0, len(index_positions)):
    pred_tag = []
    for index_info, output in zip(index_positions[i], outputs[i]):
    # label이 Mask(29)인 부분 빼고 정수를 개체명으로 변환
      if index_info != 29:
          #convert는 index to tag
        pred_tag.append(converter[output])

    pred_list.append(pred_tag)

#   print("\n-----------------------------")
  for input, preds in zip(inputs, pred_list):
    result = []
    for one_word, one_label in zip(input.split(), preds):
      result.append((one_word, one_label))
    result_list.append(result)
    # print("-----------------------------")

  return result_list


#감정사전 라벨
def label_to_index(label):
  label = label.replace(" ", "")
  if label == "불만":
    label = 0
  elif label == "중립":
    label = 1
  elif label == "당혹":
    label = 2
  elif label == "기쁨":
    label = 3
  elif label == "걱정":
    label = 4
  elif label == "질투":
    label = 5
  elif label == "슬픔":
    label = 6
  elif label == "죄책감":
    label = 7
  elif label == "연민":
    label = 8

  return label

def EMO_make_datasets(sentences, labels, max_len, tokenizer):

    input_ids, attention_masks, token_type_ids, labels_list = [], [], [], []
    tokenizer.pad_token

    for sentence, label in zip(sentences, labels):
        # 문장별로 정수 인코딩 진행
        input_id = tokenizer.encode(sentence, max_length=max_len)
        # encode한 정수들의 수만큼 1로 할당
        attention_mask = [1] * len(input_id)
        # 입력(문장)이 1개이므로 세그먼트 임베딩의 모든 차원이 0
        token_type_id = [0] * max_len

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        labels_list.append(label)

    # 패딩
    input_ids = pad_sequences(input_ids, padding='post', maxlen=max_len)
    attention_masks = pad_sequences(attention_masks, padding='post', maxlen=max_len)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)
    labels_list = np.asarray(labels_list, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), labels_list

def EMO_make_datasets_for_prediction(sentences, max_len, tokenizer):

    input_ids, attention_masks, token_type_ids = [], [], []
    tokenizer.pad_token

    for sentence in sentences:
    # 문장별로 정수 인코딩 진행
        input_id = tokenizer.encode(sentence, max_length=max_len)
        # encode한 정수들의 수만큼 1로 할당
        attention_mask = [1] * len(input_id)
        # 입력(문장)이 1개이므로 세그먼트 임베딩의 모든 차원이 0
        token_type_id = [0] * max_len

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)

    # 패딩
    input_ids = pad_sequences(input_ids, padding='post', maxlen=max_len)
    attention_masks = pad_sequences(attention_masks, padding='post', maxlen=max_len)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)

    return (input_ids, attention_masks, token_type_ids)

def emo_predict(sentences, tokenizer, model, converter, max_len=128):

    # 예측에 필요한 데이터폼 생성
    input = EMO_make_datasets_for_prediction(sentences, max_len, tokenizer)
    raw_output = model.predict(input)
    output = np.argmax(raw_output, axis=-1)

    prediction = converter[output[0]]

    return prediction
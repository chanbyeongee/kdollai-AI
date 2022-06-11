from transformers import shape_list, BertTokenizer, TFBertModel
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pickle

class TFBertForTokenClassification(tf.keras.Model):
    def __init__(self, model_name, labels):
        super(TFBertForTokenClassification, self).__init__()
        # 모델 구조 생성 (64 x 128 x 29)
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.classifier = tf.keras.layers.Dense(labels,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                                                name='classifier')
        # 개체명 사전 파일 load
        with open('C:\MyProjects\Other Projects\AI ChatBot Project\Training_Model_Weights/NER_RNN/index_to_tag.pickle', 'rb') as f:
            self.index_to_tag = pickle.load(f)
        # 사전에 만들어 놓은 BERT Tokenizer load
        self.tokenizer = get_tokenizer()

    def call(self, inputs):
        # encoding input, mask, positional encoding
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        all_output = outputs[0]
        prediction = self.classifier(all_output)

        return prediction

def get_tokenizer():
    return BertTokenizer.from_pretrained("klue/bert-base")

# 훈련 데이터셋 구조 생성 함수
def convert_examples_to_features(examples, labels, max_seq_len, tokenizer, vocab, 
                                 pad_token_id_for_segment=0, pad_token_id_for_label=29):
    # [CLS], [SEP], [PAD]
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []

    for example, label in tqdm(zip(examples, labels), total=len(examples)):
        tokens = []
        labels_ids = []
        for one_word, label_token in zip(example, label):
            # 단어들을 모두 subword로 분해
            subword_tokens = tokenizer.tokenize(one_word)
            tokens.extend(subword_tokens)
            # 정답들을 정수로 encoding 및 중간중간에 MASK(29)로 채워넣어서 정답데이터 생성
            labels_ids.extend([vocab[label_token]]+ [pad_token_id_for_label] * (len(subword_tokens) - 1))

        # 최대 126차원으로 재배열
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            labels_ids = labels_ids[:(max_seq_len - special_tokens_count)]
        
        # token은 앞뒤로 [SEP], [CLS]를, label은 [PAD]를 붙여넣음
        tokens += [sep_token]
        labels_ids += [pad_token_id_for_label]
        tokens = [cls_token] + tokens
        labels_ids = [pad_token_id_for_label] + labels_ids
        # token 정수 인코딩, attention mask 생성
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        padding_count = max_seq_len - len(input_id)
        # Padding
        input_id = input_id + ([pad_token_id] * padding_count)
        attention_mask = attention_mask + ([0] * padding_count)
        token_type_id = [pad_token_id_for_segment] * max_seq_len
        label = labels_ids + ([pad_token_id_for_label] * padding_count)

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)
        assert len(label) == max_seq_len, "Error with labels length {} vs {}".format(len(label), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        data_labels.append(label)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)
    data_labels = np.asarray(data_labels, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), data_labels

# 예측하려는 입력 문장을 BERT 입력 구조로 변환하는 함수
def convert_examples_to_features_for_prediction(examples, max_seq_len, tokenizer,
                                 pad_token_id_for_segment=0, pad_token_id_for_label=-1):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    input_ids, attention_masks, token_type_ids, label_masks = [], [], [], []

    for example in tqdm(examples):
        tokens = []
        label_mask = []
        for one_word in example:
            subword_tokens = tokenizer.tokenize(one_word)
            tokens.extend(subword_tokens)
            # label에 Mask인 부분은 -1, 정답이 있는 부분은 0으로 
            label_mask.extend([0]+ [pad_token_id_for_label] * (len(subword_tokens) - 1))

        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            label_mask = label_mask[:(max_seq_len - special_tokens_count)]

        tokens += [sep_token]
        label_mask += [pad_token_id_for_label]
        tokens = [cls_token] + tokens
        label_mask = [pad_token_id_for_label] + label_mask
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        padding_count = max_seq_len - len(input_id)
        input_id = input_id + ([pad_token_id] * padding_count)
        attention_mask = attention_mask + ([0] * padding_count)
        token_type_id = [pad_token_id_for_segment] * max_seq_len
        label_mask = label_mask + ([pad_token_id_for_label] * padding_count)

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)
        assert len(label_mask) == max_seq_len, "Error with labels length {} vs {}".format(len(label_mask), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        label_masks.append(label_mask)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)
    label_masks = np.asarray(label_masks, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), label_masks

def ner_prediction(examples, tokenizer, model, vocab, max_seq_len=128):
  examples = [sent.split() for sent in examples]
  # 입력 데이터 생성
  X_pred, label_masks = convert_examples_to_features_for_prediction(examples, max_seq_len, tokenizer)
  # 예측
  y_predicted = model.predict(X_pred)
  # 128 x 29 차원의 원핫 인코딩 형태로 확률 예측값이 나오므로 최댓값만을 뽑아내 128차원 벡터로 변환
  y_predicted = np.argmax(y_predicted, axis = -1)

  pred_list = []
  result_list = []

  for i in range(0, len(label_masks)):
    pred_tag = []
    for label_index, pred_index in zip(label_masks[i], y_predicted[i]):
    # label이 Mask(-1)인 부분 빼고 정수를 개체명으로 변환
      if label_index != -1:
        pred_tag.append(vocab[pred_index])

    pred_list.append(pred_tag)

#   print("\n-----------------------------")
  for example, pred in zip(examples, pred_list):
    one_sample_result = []
    for one_word, label_token in zip(example, pred):
      one_sample_result.append((one_word, label_token))
    #   print((one_word, label_token))
    result_list.append(one_sample_result)
    # print("-----------------------------")

  return result_list
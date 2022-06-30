from .gc_transformer import *
from .tf_bert import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import OrderedDict
from transformers import BertTokenizer
import numpy as np
import pickle
import os
## 가중치만 만들고 불러오는게 안전하다
##모델 만들어오는 함수들


class AIModel:
    emotion_labels = {"불만":0, "중립":1, "당혹":2, "기쁨":3, "걱정":4, "질투":5, "슬픔":6, "죄책감":7, "연민":8}

    emotion_mapping_by_index = dict((value, key) for (key, value) in emotion_labels.items())
    NER_mapping_by_index = {0: 'O', 1: 'PER-B', 2: 'PER-I', 3: 'FLD-B', 4: 'FLD-I', 5: 'AFW-B', 6: 'AFW-I', 7: 'ORG-B', 8: 'ORG-I', 9: 'LOC-B',
     10: 'LOC-I', 11: 'CVL-B', 12: 'CVL-I', 13: 'DAT-B', 14: 'DAT-I', 15: 'TIM-B', 16: 'TIM-I', 17: 'NUM-B',
     18: 'NUM-I', 19: 'EVT-B', 20: 'EVT-I', 21: 'ANM-B', 22: 'ANM-I', 23: 'PLT-B', 24: 'PLT-I', 25: 'MAT-B',
     26: 'MAT-I', 27: 'TRM-B', 28: 'TRM-I'}
    NER_labels = dict((value, key) for (key, value) in NER_mapping_by_index.items())


    def __init__(self):
        #self.EmotionLabel = emotion_label
        #self.mNER_tag = NER_index
        self.get_converters()

    def get_converters(self):
        self._mTokenizer = BertTokenizer.from_pretrained("klue/bert-base")

        with open(os.environ['CHATBOT_ROOT']+"/resources/converters/tokenizer.pickle", 'rb') as f:
            self._mGC_tokenizer = pickle.load(f)

    def model_loader(self):
        self.GC_model = self._load_general_corpus_model()
        self.NER_model = self._load_NER_model()
        self.EMO_model = self._load_Emo_model()

    def _load_general_corpus_model(self):
        D_MODEL = 256
        NUM_LAYERS = 2
        NUM_HEADS = 8
        DFF = 512
        DROPOUT = 0.1

        VOCAB_SIZE = self._mGC_tokenizer.vocab_size + 2
        #print(VOCAB_SIZE)

        new_model = transformer(
            vocab_size=VOCAB_SIZE,
            num_layers=NUM_LAYERS,
            dff=DFF,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dropout=DROPOUT)

        new_model.load_weights(os.environ['CHATBOT_ROOT']+"/resources/weights/Transformer_weights")

        return new_model

    def _load_NER_model(self):
        tag_size = len(AIModel.NER_labels)

        new_model = TokenClassification("klue/bert-base", labels=tag_size+1)
        new_model.load_weights(os.environ['CHATBOT_ROOT']+"/resources/weights/NER_weights")

        return new_model

    def _load_Emo_model(self):

        new_model = SequenceClassification("klue/bert-base", num_labels=len(AIModel.emotion_labels))
        new_model.load_weights(os.environ['CHATBOT_ROOT']+"/resources/weights/Emo_weights")

        return new_model


##광명님이 말하는 자료구조로 만들어주는 함수
    def run(self, name, inputsentence):
        #genral_model, NER_model, EMO-model,

        #tokenizer, GC_tokenizer, index_to_NERtag, index_to_EmotionWord, ner_labels = get_converters()

        # GeneralCorpus_model = load_general_corpus_model(GC_tokenizer)
        # NER_model = load_NER_model(ner_labels)
        # Emo_model = load_Emo_model(index_to_EmotionWord)
        self.model_loader()

        Data = OrderedDict()

        ##컨버터들 , 컨버터는 정수 to tag등...
        # GCtokenizer = GeneralCorpus_model
        # NER_converter = converters[2]
        # Emo_converter = converters[3]

        ######### classmethod로 바꾸기 VS 현행유지
        GeneralAnswer = predict(inputsentence, self._mGC_tokenizer, self.GC_model)
        #예측 함수들 모임들
        NEROut = self._ner_predict([inputsentence], self._mTokenizer)
        EmoOut = self._emo_predict([inputsentence], self._mTokenizer)

        NER = {}
        for (word, tag) in NEROut[0]:
            if tag != "O":
                NER[word] = tag

        Data["Name"] = name
        Data["Input_Corpus"] = inputsentence
        Data["NER"] = NER
        Data["Emotion"] = EmoOut
        Data["Type"] = "General"
        Data["System_Corpus"] = GeneralAnswer

        return Data


    #추후 수정 예정
    def NER_make_datasets_training(self, sentences, labels, max_len):

        input_ids, attention_masks, token_type_ids, labels_list = [], [], [], []

        for sentence, label in zip(sentences, labels):
            # 문장별로 정수 인코딩 진행
            input_id = self._mTokenizer.encode(sentence, max_length=max_len)
            # encode한 정수들의 수만큼 1로 할당
            attention_mask = [1] * len(input_id)
            # 입력(문장)이 1개이므로 세그먼트 임베딩의 모든 차원이 0
            token_type_id = [0] * max_len
            # label을 정수로 convert
            indexs = []
            for one_word, one_label in zip(sentence.split(), label):
                # label그대로 정답데이터를 만드는 것 보다, 한 단어들 모두 subword로 나뉘어서 인코딩 되므로
                # 원래 단어 위치에 맞게 label index를 넣어주고, subword로 생긴 자리에는 상관 없는 수(29)를 할당해주면서 정답데이터를 만든게 정답률이 높음
                sub_words = self._mTokenizer.tokenize(one_word)

                ############### KHUDOLL AIMODEL 75Lines tokenizer word dictionary
                indexs.extend([AIModel.NER_labels[one_label]] + [29] * (len(sub_words) - 1))

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

        # 텐서, 어텐션, 세그먼트, 답  => Training_NER_KoBERT 50번줄
        return (input_ids, attention_masks, token_type_ids), labels_list

    # 장사
    # 예측하려는 입력 문장을 BERT 입력 구조로 변환하는 함수
    # 다대일(감정)
    # 예측을 위한 함수(실제 상황에서 실시간으로 들어가는 함수)
    # 구별하는이유는 정답데이터의 유무

    def NER_make_datasets(self, sentences, max_len):
    #sentences, max_len, tokenizer
        input_ids, attention_masks, token_type_ids, index_positions = [], [], [], []

        for sentence in sentences:
            # 문장별로 정수 인코딩 진행
            input_id = self._mTokenizer.encode(sentence, max_length=max_len)
            # encode한 정수들의 수만큼 1로 할당
            attention_mask = [1] * len(input_id)
            # 입력(문장)이 1개이므로 세그먼트 임베딩의 모든 차원이 0
            token_type_id = [0] * max_len
            # label을 정수로 convert
            indexs = []
            for one_word in sentence.split():
                # 하나의 단어가 시작되는 지점을 1, subword로 생긴 자리나, pad된 부분을 29으로 표시한다. 이는 예측된 label의 자리를 나타낸 것이다.
                sub_words = self._mTokenizer.tokenize(one_word)
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

    def EMO_make_datasets(self, sentences, max_len):

        input_ids, attention_masks, token_type_ids = [], [], []

        for sentence in sentences:
            # 문장별로 정수 인코딩 진행
            input_id = self._mTokenizer.encode(sentence, max_length=max_len)
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
    #
    # 실제 예측하는 함수
    def _ner_predict(self, inputs, max_len=128):
        # inputs, tokenizer, model, converter, max_len=128
        # 입력 데이터 생성

        input_datas, index_positions = self.NER_make_datasets(inputs, max_len=max_len)
        # 예측
        raw_outputs = self.NER_model.predict(input_datas)
        # 128 x 29 차원의 원핫 인코딩 형태로 확률 예측값이 나오므로 최댓값만을 뽑아내 128차원 벡터로 변환
        outputs = np.argmax(raw_outputs, axis=-1)
        #### 감정에도 적용가능할듯

        pred_list = []
        result_list = []

        for i in range(0, len(index_positions)):
            pred_tag = []
            for index_info, output in zip(index_positions[i], outputs[i]):
                # label이 Mask(29)인 부분 빼고 정수를 개체명으로 변환
                if index_info != 29:
                    # convert는 index to tag
                    pred_tag.append(AIModel.NER_mapping_by_index[output])

            pred_list.append(pred_tag)

        #   print("\n-----------------------------")
        for input, preds in zip(inputs, pred_list):
            result = []
            for one_word, one_label in zip(input.split(), preds):
                result.append((one_word, one_label))
            result_list.append(result)
            # print("-----------------------------")

        return result_list

    def _emo_predict(self,sentences, max_len=128):

        # 예측에 필요한 데이터폼 생성
        input = self.EMO_make_datasets(sentences, max_len)
        raw_output = self.EMO_model.predict(input)
        output = np.argmax(raw_output, axis=-1)

        prediction = AIModel.emotion_mapping_by_index[output[0]]

        return prediction
    # def EMO_make_datasets(sentences, labels, max_len, tokenizer):
    #
    #     input_ids, attention_masks, token_type_ids, labels_list = [], [], [], []
    #     tokenizer.pad_token
    #
    #     for sentence, label in zip(sentences, labels):
    #         # 문장별로 정수 인코딩 진행
    #         input_id = tokenizer.encode(sentence, max_length=max_len)
    #         # encode한 정수들의 수만큼 1로 할당
    #         attention_mask = [1] * len(input_id)
    #         # 입력(문장)이 1개이므로 세그먼트 임베딩의 모든 차원이 0
    #         token_type_id = [0] * max_len
    #
    #         input_ids.append(input_id)
    #         attention_masks.append(attention_mask)
    #         token_type_ids.append(token_type_id)
    #         labels_list.append(label)
    #
    #     # 패딩
    #     input_ids = pad_sequences(input_ids, padding='post', maxlen=max_len)
    #     attention_masks = pad_sequences(attention_masks, padding='post', maxlen=max_len)
    #
    #     input_ids = np.array(input_ids, dtype=int)
    #     attention_masks = np.array(attention_masks, dtype=int)
    #     token_type_ids = np.array(token_type_ids, dtype=int)
    #     labels_list = np.asarray(labels_list, dtype=np.int32)
    #
    #     return (input_ids, attention_masks, token_type_ids), labels_list


try:

    from submodules import NER_labels, mNER_tokenizer, np, NER_mapping_by_index, index_mapping_by_NER, TokenClassification, os
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception as error:
    pass

def encode_to_integer_input(sentence):
    inputdata = []

    for letter in sentence:
        if letter not in mNER_tokenizer:
            encoded_input = 1
        else:
            encoded_input = mNER_tokenizer[letter]

        inputdata.append(encoded_input)

    return inputdata

def encode_to_integer_target(sentences, max_len):
    targetdata = []

    for sentence in sentences:
        temptarget = []
        for letter, target in sentence:
            if target not in index_mapping_by_NER:
                print("error!")
                break
            else:
                encoded_target = index_mapping_by_NER[target]

            temptarget.append(encoded_target)

        targetdata.append(temptarget)

    target_list = pad_sequences(targetdata, padding='post', maxlen=max_len, value=13)

    target_list = np.asarray(target_list, dtype=np.int32)

    return target_list

def ner_make_datasets(sentences, max_len, tokenizer):

    cls_index = tokenizer['[CLS]']
    sep_index = tokenizer['[SEP]']

    input_ids, attention_masks, token_type_ids = [], [], []

    for sentence in sentences:
        input_id = encode_to_integer_input(sentence)
        input_id = input_id + [sep_index]
        input_id = [cls_index] + input_id
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

# def ner_make_datasets_training(sentences, labels, max_len):
#     input_ids, attention_masks, token_type_ids, labels_list = [], [], [], []

#     for sentence, label in zip(sentences, labels):
#         # 문장별로 정수 인코딩 진행
#         input_id = mTokenizer.encode(sentence, max_length=max_len)
#         # encode한 정수들의 수만큼 1로 할당
#         attention_mask = [1] * len(input_id)
#         # 입력(문장)이 1개이므로 세그먼트 임베딩의 모든 차원이 0
#         token_type_id = [0] * max_len
#         # label을 정수로 convert
#         indexs = []
#         for one_word, one_label in zip(sentence.split(), label):
#             # label그대로 정답데이터를 만드는 것 보다, 한 단어들 모두 subword로 나뉘어서 인코딩 되므로
#             # 원래 단어 위치에 맞게 label index를 넣어주고, subword로 생긴 자리에는 상관 없는 수(29)를 할당해주면서 정답데이터를 만든게 정답률이 높음
#             sub_words = mTokenizer.tokenize(one_word)

#             ############### KHUDOLL AIMODEL 75Lines tokenizer word dictionary
#             indexs.extend([NER_labels[one_label]] + [29] * (len(sub_words) - 1))

#         indexs = indexs[:max_len]

#         input_ids.append(input_id)
#         attention_masks.append(attention_mask)
#         token_type_ids.append(token_type_id)
#         labels_list.append(indexs)

#     # 패딩
#     input_ids = pad_sequences(input_ids, padding='post', maxlen=max_len)
#     attention_masks = pad_sequences(attention_masks, padding='post', maxlen=max_len)
#     labels_list = pad_sequences(labels_list, padding='post', maxlen=max_len, value=29)

#     input_ids = np.array(input_ids, dtype=int)
#     attention_masks = np.array(attention_masks, dtype=int)
#     token_type_ids = np.array(token_type_ids, dtype=int)
#     labels_list = np.asarray(labels_list, dtype=np.int32)

#     # 텐서, 어텐션, 세그먼트, 답  => Training_NER_KoBERT 50번줄
#     return (input_ids, attention_masks, token_type_ids), labels_list


# def ner_make_datasets(sentences, max_len):

#     input_ids, attention_masks, token_type_ids, index_positions = [], [], [], []

#     for sentence in sentences:
#         # 문장별로 정수 인코딩 진행


#         input_id = mTokenizer.encode(sentence,max_length=max_len)
#         #maxlength
#         # encode한 정수들의 수만큼 1로 할당

#         attention_mask = [1] * len(input_id)
#         # 입력(문장)이 1개이므로 세그먼트 임베딩의 모든 차원이 0
#         token_type_id = [0] * max_len
#         # label을 정수로 convert
#         indexs = []
#         for one_word in sentence.split():
#             # 하나의 단어가 시작되는 지점을 1, subword로 생긴 자리나, pad된 부분을 29으로 표시한다. 이는 예측된 label의 자리를 나타낸 것이다.
#             sub_words = mTokenizer.tokenize(one_word)
#             indexs.extend([1] + [29] * (len(sub_words) - 1))


#         indexs = indexs[:max_len]

#         input_ids.append(input_id)
#         attention_masks.append(attention_mask)
#         token_type_ids.append(token_type_id)
#         index_positions.append(indexs)


#     # 패딩
#     input_ids = pad_sequences(input_ids, padding='post', maxlen=max_len)
#     attention_masks = pad_sequences(attention_masks, padding='post', maxlen=max_len)
#     index_positions = pad_sequences(index_positions, padding='post', maxlen=max_len, value=29)


#     input_ids = np.array(input_ids, dtype=int)
#     attention_masks = np.array(attention_masks, dtype=int)
#     token_type_ids = np.array(token_type_ids, dtype=int)
#     index_positions = np.asarray(index_positions, dtype=np.int32)


#     # index_positions은 1이면 정답이 있을곳 29면 아님
#     return (input_ids, attention_masks, token_type_ids), index_positions

def ner_predict(model, inputs, max_len=128):
    # inputs, tokenizer, model, converter, max_len=128
    # 입력 데이터 생성

    input_datas = ner_make_datasets(inputs, max_len=max_len, tokenizer=mNER_tokenizer)
    # 예측

    raw_outputs = model.predict(input_datas)
    # 128 x 29 차원의 원핫 인코딩 형태로 확률 예측값이 나오므로 최댓값만을 뽑아내 128차원 벡터로 변환
    y_predicted = np.argmax(raw_outputs, axis=-1)
    #### 감정에도 적용가능할듯
    
    LC, QT, OG, DT, PS, TI = "", "", "", "", "", ""

    LClist, QTlist, OGlist, DTlist, PSlist, TIlist = [], [], [], [], [], []

    result_list = []

    for i in range(len(inputs)):
        if y_predicted[0][i] == 1:
            LC = inputs[i]
            while y_predicted[0][i+1] == 2:
                LC += inputs[i+1]
                i += 1
            if LC not in result_list: result_list.append(("LC", LC))
            LC = ""
        elif y_predicted[0][i] == 3:
            QT = inputs[i]
            while y_predicted[0][i+1] == 4:
                QT += inputs[i+1]
                i += 1
            if QT not in result_list: result_list.append(("QT", QT))
            QT = ""
        elif y_predicted[0][i] == 5:
            OG = inputs[i]
            while y_predicted[0][i+1] == 6:
                OG += inputs[i+1]
                i += 1
            if OG not in result_list: result_list.append(("OG", OG))
            OG = ""
        elif y_predicted[0][i] == 7:
            DT = inputs[i]
            while y_predicted[0][i+1] == 8:
                DT += inputs[i+1]
                i += 1
            if DT not in result_list: result_list.append(("DT", DT))
            DT = ""
        elif y_predicted[0][i] == 9:
            PS = inputs[i]
            while y_predicted[0][i+1] == 10:
                PS += inputs[i+1]
                i += 1
            if PS not in result_list: result_list.append(("PS", PS))
            PS = ""
        elif y_predicted[0][i] == 11:
            TI = inputs[i]
            while y_predicted[0][i+1] == 12:
                TI += inputs[i+1]
                i += 1
            if TI not in result_list: result_list.append(("TI", TI))
            TI = ""

    # pred_list = []
    # result_list = []

    # for i in range(0, len(index_positions)):
    #     pred_tag = []
    #     for index_info, output in zip(index_positions[i], outputs[i]):
    #         # label이 Mask(29)인 부분 빼고 정수를 개체명으로 변환
    #         if index_info != 29:
    #             # convert는 index to tag
    #             pred_tag.append(NER_mapping_by_index[output])

    #     pred_list.append(pred_tag)

    # #   print("\n-----------------------------")
    # for input, preds in zip(inputs, pred_list):
    #     result = []
    #     for one_word, one_label in zip(input.split(), preds):
    #         result.append((one_word, one_label))
    #     result_list.append(result)
    #     # print("-----------------------------")
    return result_list

def load_NER_model():
    print("########Loading NER model!!!########")
    tag_size = len(NER_labels)
    new_model = TokenClassification("klue/bert-base", labels=tag_size+1)
    new_model.load_weights(os.environ['CHATBOT_ROOT']+"/resources/weights/NER_weights/NER_weights")

    return new_model

new_model = load_NER_model()

sample = "오늘 엄마가 심하게 때렸어"

prediction = ner_predict(new_model, sample)
print(prediction)
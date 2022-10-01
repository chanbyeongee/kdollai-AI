"""
last modified : 220925
modified by : Heo_Yoon
contents : (new)topic_classifier
new_dependencies(ex. lib) : (new_folder)resources/the_weights
                            , (new lib) eunjeon, gensim
"""

from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath
from eunjeon import Mecab
import os
import numpy as np

main_topic_label = {0: "가족", 1: "학교", 2: "건강", 3: "취미", 4: "취미", 5: "취미",
                    6: "방송_미디어", 7: "방송_미디어", 8: "날씨_및_계절", 9: "반려동물", 10: "식음료"}

sub_topic_label = {0: None, 1: None, 2: None, 3: "스포츠_레저", 4: "여행", 5: "게임", 6: "방송_연예",
                   7: "영화_만화", 8: None, 9: None, 10: None}

# model이 출력하는 topic index를 label의 index로 바꾸어 줍니다.
topic_index_dict = {0: 10, 2: 8, 6: 9, 10: 1}
hobby_index_dict = {0: 5, 1: 3, 2: 4}
broad_media_dict = {0: 6, 1: 7}
family_health_dict = {0: 0, 1: 2}

def Topic_predict(LDA_model, sentences):
    # 문장을 형태소 단위로 분리
    m = Mecab()
    key = m.pos(sentences)
    key_list = []
    # 품사가 명사, 동사 이며 1글자 이상인 단어만 취함
    for L, pos in key:
        if pos.startswith("N") or pos.startswith("V"):
            if len(L) > 1:
                key_list.append(L)
    # vocab 비교
    bow = LDA_model[0].id2word.doc2bow(key_list)
    # 일치 단어 없을 시 무주제
    if not bow:  #
        return "무주제"
    # 일치 단어 있을 시 model 입력
    topic_distribution = LDA_model[0].get_document_topics(bow)
    print(topic_distribution)
    temp1 = []  # topic number 저장
    temp2 = []  # topic probability 저장
    for num, prob in topic_distribution:
        temp1.append(num)
        temp2.append(prob)
    # 가장 높게 예측된 주제 index
    topic_num = temp1[np.argmax(temp2)]
    if np.max(temp2) < 0.85:  # 예상 확률이 90% 미만일 때
        return None, None
    else:
        if (topic_num == 4) or (topic_num == 5) or (topic_num == 8):  # 취미로 판정될 경우
            bow2 = LDA_model[1].id2word.doc2bow(key_list)
            if not bow2:
                return None, None
            topic_distribution = LDA_model[1].get_document_topics(bow2)
            temp1 = []
            temp2 = []
            for i, j in topic_distribution:
                temp1.append(i)
                temp2.append(j)
            topic_num = temp1[np.argmax(temp2)]
            index = hobby_index_dict[topic_num]

        elif (topic_num == 3) or (topic_num == 7):
            bow3 = LDA_model[2].id2word.doc2bow(key_list)
            if not bow3:
                return None, None
            topic_distribution = LDA_model[2].get_document_topics(bow3)
            temp1 = []
            temp2 = []
            for i, j in topic_distribution:
                temp1.append(i)
                temp2.append(j)
            topic_num = temp1[np.argmax(temp2)]
            index = broad_media_dict[topic_num]

        elif topic_num == 1:
            bow4 = LDA_model[3].id2word.doc2bow(key_list)
            if not bow4:
                return None, None
            topic_distribution = LDA_model[3].get_document_topics(bow4)
            temp1 = []
            temp2 = []
            for i, j in topic_distribution:
                temp1.append(i)
                temp2.append(j)
            topic_num = temp1[np.argmax(temp2)]
            index = family_health_dict[topic_num]
            print(index)
        else:
            index = topic_index_dict[topic_num]

        if np.max(temp2) > 0.9:
            main_topic = main_topic_label[index]
            sub_topic = sub_topic_label[index]
            return main_topic, sub_topic
        else:
            return None, None


def load_Topic_model():
    print("########Loading THE model!!!########")
    save_dir = datapath(os.environ['CHATBOT_ROOT'] + "/resources/weights/Topic_weights/")  # 토픽 모델 저장소
    main_model = LdaModel.load(save_dir + "LDA_model_topic11(Hanieum)")
    sub_hobby_model = LdaModel.load(save_dir + "LDA_hobby")
    sub_media_model = LdaModel.load(save_dir + "LDA_media")
    sub_family_health_model = LdaModel.load(save_dir + "LDA_family_health")

    return main_model, sub_hobby_model, sub_media_model, sub_family_health_model

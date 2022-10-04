""""
last modified : 220925
modified by : Heo Yoon
contents : (new) topic_classfier using LDA, solve the conflict
new_dependencies : (module) the_classifier
"""
try :
    from setup import setup_environ
    import os

    setup_environ()
    # CPU만 사용
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # GPU log 설정
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from submodules import *
    from ToolPack.danger_detector import *
    from collections import OrderedDict

except Exception :
    from .submodules import *
    from .ToolPack.danger_detector import *
    from collections import OrderedDict
    pass

## device 관련 설정

## 가중치만 만들고 불러오는게 안전하다
##모델 만들어오는 함수들

class AIModel:

    def __init__(self):
        self.get_converters()
        self.dialog_buffer = []

    def get_converters(self):
        self._mTokenizer = mTokenizer
        self._mGC_tokenizer = mGC_tokenizer

    def model_loader(self):
        self.GC_model = load_general_corpus_model()
        self.NER_model = load_NER_model()
        self.EMO_model = load_Emo_model()
        self.Topic_model = load_Topic_model()
        self.danger_detector = detector()

    def manage_dailogbuffer(self):
        if len(self.dialog_buffer) < 3:
            return False

        elif len(self.dialog_buffer) == 3:
            return True

        else:
            while len(self.dialog_buffer) != 3:
                self.dialog_buffer.pop(0)
            return True


    def get_results(self, inputsentence):
        dialogs = ""
        for dialog in self.dialog_buffer:
            dialogs += " " + dialog

        GeneralAnswer = GD_predict(inputsentence, self.GC_model, self._mTokenizer)
        NEROut = ner_predict(self.NER_model,[inputsentence])
        EmoOut = emo_predict(self.EMO_model,[inputsentence])

        NER = {}
        for (word, tag) in NEROut:
            NER[word] = tag

        # print(len(self.dialog_buffer))

        if self.manage_dailogbuffer() is True:
            (main_topic, sub_topic) = Topic_predict(self.Topic_model, dialogs, EmoOut)
            print(dialogs)
        else:
            main_topic = None
            sub_topic = None

        if main_topic in ["가족", "건강", "학교"]:
            TypeOut = "Scenario"
        else:
            TypeOut = "General"

        return GeneralAnswer, NER, EmoOut, main_topic, sub_topic, TypeOut

# 분석은 더미데이터로

# 감정 시나리오
# 6개 정도 가다가

# 주제 시나리오
#

## 나 오늘 너무 슬퍼
# 무슨일 있어?, 더 자세히말해줄래, 슬프겠다... 더 말해줄수있어?
# 나 오늘 친구랑 싸웠어
# 유저 인풋 3가지
#


##광명님이 말하는 자료구조로 만들어주는 함수
    def run(self, name, inputsentence):

        Data = OrderedDict()
        self.dialog_buffer.append(inputsentence)

        GeneralAnswer, Name_Entity, Emotion, main_topic, sub_topic, TypeOut = self.get_results(inputsentence)

        DangerFlag, Badwords = self.danger_detector.detect(inputsentence)

        Data["Name"] = name
        Data["Input_Corpus"] = inputsentence
        Data["NER"] = Name_Entity
        Data["Emotion"] = Emotion
        Data["Topic"] = main_topic
        Data["Sub_Topic"] = sub_topic
        Data["Type"] = TypeOut
        Data["System_Corpus"] = GeneralAnswer
        Data["Danger_Flag"] = DangerFlag
        Data["Danger_Words"] = Badwords

        return Data

if __name__ == "__main__":

    DoDam = AIModel()

    DoDam.model_loader()

    UserName = "민채"

    while True:
        sample = input("입력 : ")
        output = DoDam.run(UserName, sample)
        print("출력 : {}" .format(output))



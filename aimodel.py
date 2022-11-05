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
    cnt = 0
    state = "general"

    def __init__(self):
        self.get_converters()
        self.dialog_buffer = []
        self.model_loader()
        emo_predict(self.EMO_model, ["안녕"])
        Topic_predict(self.Topic_model, "안녕", "기쁨")
        ner_predict(self.NER_model, "[안녕]")
        GD_predict("안녕", self.GC_model, self._mTokenizer)
        self.danger_detector.detect("안녕")
        yes_no_predict(self.yes_no_model, "안녕")

    def set_init(self):
        self.cnt = 0
        self.state = "general"

    def get_converters(self):
        self._mTokenizer = mTokenizer
        self._mGC_tokenizer = mGC_tokenizer

    def model_loader(self):
        self.GC_model = load_general_corpus_model()
        self.NER_model = load_NER_model()
        self.EMO_model = load_Emo_model()
        self.Topic_model = load_Topic_model()
        self.yes_no_model = load_yes_no_model()
        self.danger_detector = detector()

    def manage_dailogbuffer(self):
        if len(self.dialog_buffer) < 1:
            return False

        elif len(self.dialog_buffer) == 1:
            return True

        else:
            while len(self.dialog_buffer) != 1:
                self.dialog_buffer.pop(0)
            return True


    def get_results(self, name, inputsentence):
        EmoOut = None
        dialogs = ""
        for dialog in self.dialog_buffer:
            dialogs += dialog

        if self.cnt < 2:
            EmoOut = emo_predict(self.EMO_model, [inputsentence])

        if self.manage_dailogbuffer() is True:
            topic = Topic_predict(self.Topic_model, dialogs, EmoOut)
        else:
            topic = None

        if self.cnt == 0 and EmoOut in ["당혹", "죄책감", "슬픔", "연민", "걱정", "기쁨", "불만", "질투"]:
            if EmoOut == "슬픔" and topic == "가족":
                self.state = "가족_슬픔"
            else:
                self.state = EmoOut
            self.cnt = 1
            self.s_flag = True

        if self.state == "general":
            TypeOut = "General"
            GeneralAnswer = [GD_predict(inputsentence, self.GC_model, self._mTokenizer)]


        else:  # 당혹, 죄책감, 슬픔, 연민, 걱정, 기쁨
            TypeOut = "Scenario"

            if self.cnt == 2:
                self.s_flag = False

            if self.state == "가족_슬픔":
                if self.cnt == 1:
                    GeneralAnswer = ["그래...? 부모님께서 싸우셨다고 말한거지?"]
                    self.cnt += 1


                elif self.cnt == 2:
                    GeneralAnswer = ["그렇구나... 부모님이 자주 싸우시니?"]
                    self.cnt += 1

                elif self.cnt == 3:
                    GeneralAnswer = ["그럼 동현이는 부모님이 싸우는 걸 보면 어떤 생각이 들어?"]
                    self.cnt += 1

                elif self.cnt == 4:
                    GeneralAnswer = ["나도 동현이 마음이 이해돼... "+
                                     "부모님께서 조금만 덜 싸우면 좋겠는데... 내가 위로해 주고 싶어 "+
                                     "그일 때문에 많이 슬퍼보이는데 맞니?"]
                    self.cnt += 1

                elif self.cnt == 5:
                    reaction = yes_no_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        self.state = "슬픔"
                        self.cnt = 2
                        GeneralAnswer = self.get_results(name, inputsentence)[0]
                    else:
                        GeneralAnswer  = ["내 감이 틀렸다니 다행이다, 그래도 내 도움이 필요하면 꼭 말해줘! "+
                                          "항상 널 응원할게"]
                        self.cnt = 0
                        self.state = "general"

            elif self.state == "당혹":
                if self.cnt == 1:
                    GeneralAnswer = ["음.. 오늘 " + name + "에게 당황스러운 일이 있었나보구나.. "+
                                        "괜찮다면 어떤 일이 있었는지 물어봐도 되니?"]
                    self.cnt += 1


                elif self.cnt == 2:
                    GeneralAnswer = ["저런.. 내가 너였어도 많이 당황스러웠을 거 같아.. "+
                                     "많이 놀랐을텐데 나에게 얘기해줘서 정말 고마워! "+
                                     "이 얘기에 대해서 너랑 이야기를 더 해보고싶은데 괜찮을까?"]
                    self.cnt += 1

                elif self.cnt == 3:
                    reaction = yes_no_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["고마워! 얘기를 나누면서 내가 너에게 도움이 되었으면 좋겠다.. "+
                                         "그 전에 너를 당황스럽게 한 그 상황에 대해 자세히 알고 싶은데.. 혹시 뭔지 알 수 있을까?"]
                        self.cnt += 1
                    else:
                        GeneralAnswer = ["내가 너무 성급했나봐.. 너한테 너무 부담을 준 거 같아서 정말 미안해.. "+
                                         "다음에라도 얘기해줄 수 있다면 언제든지 찾아와줘! "+
                                         "그동안 더 많이 배워서 " + name + "한테 꼭 도움을 주고 싶어. 오늘 나랑 얘기해줘서 고마워!"]
                        self.cnt = 0
                        self.state = "general"

                elif self.cnt == 4:
                    GeneralAnswer = ["말해줘서 정말 고마워! "+
                                     "내가 알기로는, 사람이 당황스럽다고 느껴지는 상황을 마주하게 되면 크게 위축된다고 해. "+
                                     "아마 그런 상황 때문에 너가 침착하게 생각할 수 없었던 거 일지도 몰라. "+
                                     "너는 당황스러움을 벗어나기 위해서 어떤 걸 하니?"]
                    self.cnt += 1
                elif self.cnt == 5:
                    GeneralAnswer = ["오! 정말 좋은 방법인 걸? "+
                                     "내가 너의 지친 마음이 괜찮아질 때까지 옆에서 계속 응원해줄게! "+
                                     "내가 생각나면 언제는 나를 찾아줘! 다음에 또 보자!"]
                    self.cnt = 0
                    self.state = "general"

            elif self.state == "죄책감":
                if self.cnt == 1:
                    GeneralAnswer = ["가볍게 생각할 수 있지만 너가 지금 느끼는 감정은 어쩌면 너를 정말 힘들게 만들지도 몰라... "+
                                     "어째서 그렇게 생각한 건지 더 자세히 말해줄 수 있니?"]
                    self.cnt += 1
                elif self.cnt == 2:
                    GeneralAnswer = ["그렇구나... 말해줘서 정말 고마워.. "+
                                    "보통 이런 상황에서는 내 편이 없다고 느끼기 쉽고, 실제로도 없어서 힘든 경우가 많은 걸로 알고 있어.. "+
                                    "너는 지금 어떻니? 지금 혼자라는 생각이 드니?"]
                    self.cnt += 1
                elif self.cnt == 3:
                    reaction = yes_no_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["그렇구나.. 계속 혼자서 이런 상황을 버틴거구나... "+
                                         "너가 혼자 힘들게 버텼을 생각을 하니 나도 슬프다.. "+
                                         "누군가한테 너는 정말 소중한 사람이라는 걸 잊지 말아줘."]
                    else:
                        GeneralAnswer = ["휴.. 그래도 정말 다행이야..! "+
                                         "이런 힘든 상황에서 나를 지지해줄 사람이 있다는 건 정말이지 큰 힘이니까 말이야.. "+
                                         "그래도 너무 힘들어하지 않았으면 좋겠어.. 누군가한테 너는 정말 소중한 사람이라는 걸 잊지 말아줘."]
                    self.cnt = 0
                    self.state = "general"

            elif self.state == "슬픔":
                if self.cnt == 1:
                    GeneralAnswer = ["음.. 이야기를 들으면서 느낀건데, 너 지금 많이 슬퍼하는 것 같어. 내 생각이 맞니? "+
                                     "맞다면 무슨 일이 있던건지 더 자세히 말해줄래?"]
                    self.cnt += 1
                    print(self.cnt)

                elif self.cnt == 2:
                    GeneralAnswer = ["그랬구나.. 말해줘서 정말 고마워. "+
                                     "너가 슬퍼하는 것 같아서 나도 마음이 너무 아프다.. "+
                                     "혹시 슬픈 상황을 견디는 너만의 방법이 있니?"]
                    self.cnt += 1

                elif self.cnt == 3:
                    reaction = yes_no_predict(self.yes_no_model, inputsentence)
                    print(reaction)
                    if reaction == "yes":
                        GeneralAnswer = ["오, 정말 다행이다. "+
                                         "슬픔을 너가 잘 조절할 수 있다면 그 슬픔으로 인해 오히려 너가 더 성장할 수 있다고 해. "+
                                         "그러니 앞으로도 슬픈 일이 있을 때 너만의 방법으로 잘 극복했으면 좋겠어. 다음에 또 봐!"]

                    else:
                        GeneralAnswer = ["그렇구나.. "+
                                         "그렇다면 노래를 한 번 들어보는 건 어떻니? 아이유 가수의 밤 편지라는 노래가 너에게 위로를 줄 것 같은데, 한 번 들어줬으면 좋겠어. "+
                                         "아무튼, 긴 얘기 들어줘서 고마워! 다음에 또 봐!"]
                    self.cnt = 0
                    self.state = "general"

            elif self.state == "연민":
                if self.cnt == 1:
                    GeneralAnswer = ["그래? 계속 신경이 쓰이겠다 " + name + "이가. 혹시 더 자세히 말해줄 수 있어?"]
                    self.cnt += 1

                elif self.cnt == 2:
                    GeneralAnswer = ["그랬구나. 말해줘서 고마워. 내가 이야기를 들어보니까 " + name + "이한테 해주고 싶은 말이 생겼는데 들어볼래?"]
                    self.cnt += 1

                elif self.cnt == 3:
                    reaction = yes_no_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["나는 " + name + "이가 신경써주는게 정말 멋진 일인 것 같아. "+
                                         "그 사람 이야기를 계속 들어주는 것만으로도 되게 큰 위로가 될거야! 그리고 예쁜 말해주고 안아주면 더 좋아할 걸? "+
                                         "근데 그래도 너무 스트레스 받아하지는 마. 그 사람이 힘들 수는 있지만 그게 " + name + "이 탓은 아니니까. 내 말 알겠지? "+
                                         "같이 용기내보자! 내가 응원할게."]
                    else:
                        GeneralAnswer = ["그렇구나. 알겠어 다음에 얘기하고 싶으면 말해주라."]

                    self.cnt = 0
                    self.state = "general"

            elif self.state == "걱정":
                if self.cnt == 1:
                    GeneralAnswer = ["내 생각엔 " + name + "이가 걱정을 하고 있는 것 같네. 맞다면 무슨 일인지 더 자세히 말해줄래?"]
                    self.cnt += 1

                elif self.cnt == 2:
                    GeneralAnswer = ["그런 일이었구나.. 말해줘서 고마워. 내가 마음이 조금 나아질 수 있는 방법을 아는데 알려줄까?"]
                    self.cnt += 1

                elif self.cnt == 3:
                    reaction = yes_no_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["지금 " + name + "이가 느끼는게 지금은 많이 걱정스러운 것 처럼 보이지만 사실은 아닐 수도 있어. "+
                                         "별일 아니야. 괜찮아! 하다보면 정말 괜찮아지는게 대부분이다! 별거 아닌걸로 걱정했구나 하는 경우가 많대. "+
                                         "다 잘 될거야. 내가 옆에서 지켜줄게."]
                    else:
                        GeneralAnswer = ["그렇구나. 알겠어 다음에 얘기하고 싶으면 말해주라."]
                    self.cnt = 0
                    self.state = "general"

            elif self.state == "기쁨":
                if self.cnt == 1:
                    GeneralAnswer = ["지금 " + name + "이가 기분이 좋은 것 같네! 그 얘기 좀 더 자세히 해주라."]
                    self.cnt += 1

                elif self.cnt == 2:
                    GeneralAnswer = ["그렇구나! 말해줘서 고마워. " + name + "이가 기분 좋아서 나도 좋다. "+
                                     "나도 " + name + "이한테 어떤 얘기 해주고 싶은데 들어볼래?"]
                    self.cnt += 1
                elif self.cnt == 3:
                    reaction = yes_no_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["오늘 있었던 기분 좋은 일을 일기에 써보는 건 어때? 나중에 " + name + "이가 힘들고 슬플 때 읽어보면 위로도 되고 기분이 나이질 지도 몰라!"]
                    else:
                        GeneralAnswer = ["그렇구나. 알겠어 계속 기분 좋은 일만 일어났으면 좋겠다!"]
                    self.cnt = 0
                    self.state = "general"

            elif self.state == "불만":
                if self.cnt == 1:
                    GeneralAnswer = [name + "이가 지금 화가 난 것 같네. "+
                                     "진정하고 내 얘기 좀 들어볼래?"]
                    self.cnt += 1

                elif self.cnt == 2:
                    reaction = yes_no_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["난 " + name + "이가 화낼 수 있다 생각해. 어떻게 사람이 맨날 참니? 하지만 화를 낼 때도 멋지게 화를 내야 해 "+
                                         "울거나 화를 내지 말고 " + name + "이가 왜 불만이 생겼고, 앞으로 이렇게 해달라 분명하게 말하는게 중요해. 그러면 귀기울여 마음을 들어줄거야 "+
                                         "내가 " + name + "이를 응원할테니 한 번 용기를 내봐!"]
                    else:
                        GeneralAnswer = ["그렇구나. 알겠어 다음에 얘기하고 싶으면 말해주라."]

                    self.cnt = 0
                    self.state = "general"

            else:
                if self.cnt == 1:
                    GeneralAnswer = ["내 생각에는 지금 " + name + "이가 조금 질투를 하는 거 같아."+
                                     "혹시 내 얘기를 들어볼래?"]
                elif self.cnt == 2:
                    reaction = yes_no_predict(self.yes_no_model, inputsentence)
                    if reaction == "yes":
                        GeneralAnswer = ["우리는 맨날 다른 사람을 비교하고 비교 당하는 거 같아."+
                                         "하지만 " + name + "이도 세상에서 하나밖에 없는 이쁜 아이니까 누군가를 질투하지 않아도 돼!"+
                                         "내가 " + name + "이가 누구보다 멋지다는 걸 기억하고 있을게!"+
                                         "다음에도 그런 마음이 들면 나한테 또 말해줘."]
                    else:
                        GeneralAnswer = ["그렇구나. 알겠어 다음에 얘기하고 싶으면 말해주라."]

                    self.cnt = 0
                    self.state = "general"

        NEROut = ner_predict(self.NER_model, [inputsentence])
        NER = {}
        for (word, tag) in NEROut:
            NER[word] = tag


        # if main_topic in ["가족", "건강", "학교"]:
        #     TypeOut = "Scenario"
        # else:
        #     TypeOut = "General"

        return GeneralAnswer, NER, EmoOut, topic, TypeOut, None



##광명님이 말하는 자료구조로 만들어주는 함수
    def run(self, name, inputsentence):

        Data = OrderedDict()
        self.dialog_buffer.append(inputsentence)

        GeneralAnswer, Name_Entity, Emotion, topic, TypeOut, Flag = self.get_results(name, inputsentence)

        DangerFlag, Badwords = self.danger_detector.detect(inputsentence)

        Data["Name"] = name
        Data["Input_Corpus"] = inputsentence
        Data["NER"] = Name_Entity
        Data["Emotion"] = Emotion
        Data["Topic"] = topic
        Data["Type"] = TypeOut
        Data["System_Corpus"] = GeneralAnswer
        Data["Flag"] = Flag
        Data["Danger_Flag"] = DangerFlag
        Data["Danger_Words"] = Badwords

        return Data

if __name__ == "__main__":

    DoDam = AIModel()


    UserName = "민채"

    while True:
        sample = input("입력 : ")
        output = DoDam.run(UserName, sample)
        print("출력 : {}" .format(output))



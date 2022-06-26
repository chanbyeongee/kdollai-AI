from models.transformers.gc_transformer import *
from models.transformers.bertmodels.AIKoBERT import ner_predict, emo_predict
import pickle
from collections import OrderedDict
from transformers import BertTokenizer


## 가중치만 만들고 불러오는게 안전하다
##모델 만들어오는 함수들

class AIModel:
    def __init__(self):
        self._mTokenizer, self._mGC_tokenizer, self._mNER_tag, self._mEmotion_word, self._mNer_labels = self.get_converters()

    def model_loader(self):
        self.GC_model = self.load_general_corpus_model()
        self.NER_model = self.load_NER_model()
        self.EMO_model = self.load_Emo_model()

    def load_general_corpus_model(self):
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

        new_model.load_weights("C:\MyProjects\Other Projects\AI ChatBot Project\Training_Model_Weights\Transformer_chatbot3\Chatbot_Transformer3_weights")

        return new_model

    def load_NER_model(self):
        tag_size = len(self._mNer_labels)

        new_model = TFBertForTokenClassification("klue/bert-base", labels=tag_size+1)
        new_model.load_weights("C:\MyProjects\Other Projects\AI ChatBot Project\Training_Model_Weights/NER_KlueBERT_tuning/NER_KoBERT")

        return new_model

    def load_Emo_model(self):
        path = "C:\MyProjects\Other Projects\AI ChatBot Project\Training_Model_Weights\EmoClassification_KlueBERT"

        new_model = TFBertForSequenceClassification("klue/bert-base", num_labels=len(index_to_EmotionWord))
        new_model.load_weights(path+"\EmoClass_KoBERT")

        return new_model


##광명님이 말하는 자료구조로 만들어주는 함수
    def To_DataStructure(self,input):
        #genral_model, NER_model, EMO-model,

        tokenizer, GC_tokenizer, index_to_NERtag, index_to_EmotionWord, ner_labels = get_converters()

        # GeneralCorpus_model = load_general_corpus_model(GC_tokenizer)
        # NER_model = load_NER_model(ner_labels)
        # Emo_model = load_Emo_model(index_to_EmotionWord)
        self.model_loader()

        Data = OrderedDict()
        sentence = input[0]
        name = input[1]

        ##컨버터들 , 컨버터는 정수 to tag등...
        # GCtokenizer = GeneralCorpus_model
        # NER_converter = converters[2]
        # Emo_converter = converters[3]

        ######### classmethod로 바꾸기 VS 현행유지
        GeneralAnswer = predict(sentence, self._mGC_tokenizer, self.GC_model)
        #예측 함수들 모임들
        NEROut = ner_predict([sentence], self._mTokenizer, self.NER_model, self._mNER_tag)
        EmoOut = emo_predict([sentence], self._mTokenizer, self.EMO_model, self._mEmotion_word)

        NER = {}
        for (word, tag) in NEROut[0]:
            if tag != "O":
                NER[word] = tag

        Data["Name"] = name
        Data["Input_Corpus"] = sentence
        Data["NER"] = NER
        Data["Emotion"] = EmoOut
        Data["Type"] = "General"
        Data["System_Corpus"] = GeneralAnswer

        return Data

    def get_converters(self):
        path = 'C:\MyProjects\Other Projects\AI ChatBot Project\Training_Model_Weights'
        tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
        with open(path+'/Transformer_chatbot3/tokenizer.pickle', 'rb') as f:
            GC_tokenizer = pickle.load(f)
        with open(path+'/NER_RNN/index_to_tag.pickle', 'rb') as f:
            index_to_NERtag = pickle.load(f)
        with open(path+'\EmoClassification_KlueBERT\Index_to_EmotionWord.pickle', 'rb') as f:
            index_to_EmotionWord = pickle.load(f)
        ner_labels = [label.strip() for label in open('C:\MyProjects\Other Projects\AI ChatBot Project/ner_label.txt', 'r', encoding='utf-8')]

        return (tokenizer, GC_tokenizer, index_to_NERtag, index_to_EmotionWord, ner_labels)

from AITransformer import transformer, predict
from AIKoBERT import TFBertForTokenClassification, TFBertForSequenceClassification, ner_predict, emo_predict
import pickle
from collections import OrderedDict
from transformers import shape_list, BertTokenizer


## 가중치만 만들고 불러오는게 안전하다


##모델 만들어오는 함수들

def load_general_corpus_model(tokenizer):
    D_MODEL = 256
    NUM_LAYERS = 2
    NUM_HEADS = 8
    DFF = 512
    DROPOUT = 0.1

    VOCAB_SIZE = tokenizer.vocab_size + 2
    print(VOCAB_SIZE)

    new_model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        dff=DFF,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)

    new_model.load_weights("C:\MyProjects\Other Projects\AI ChatBot Project\Training_Model_Weights\Transformer_chatbot3\Chatbot_Transformer3_weights")

    return new_model

def load_NER_model(labels):
    tag_size = len(labels)

    new_model = TFBertForTokenClassification("klue/bert-base", labels=tag_size+1)

    new_model.load_weights("C:\MyProjects\Other Projects\AI ChatBot Project\Training_Model_Weights/NER_KlueBERT_tuning/NER_KoBERT")

    return new_model

def load_Emo_model(converter):
    path = "C:\MyProjects\Other Projects\AI ChatBot Project\Training_Model_Weights\EmoClassification_KlueBERT"

    new_model = TFBertForSequenceClassification("klue/bert-base", num_labels=len(converter))
    new_model.load_weights(path+"\EmoClass_KoBERT")

    return new_model


##광명님이 말하는 자료구조로 만들어주는 함수
def To_DataStructure(input, general_model, NER_model, Emo_model, converters):
    Data = OrderedDict()
    sentence = input[0]
    name = input[1]

    ##컨버터들 , 컨버터는 정수 to tag등...
    tokenizer = converters[0]
    GCtokenizer = converters[1]
    NER_converter = converters[2]
    Emo_converter = converters[3]

    GeneralAnswer = predict(sentence, GCtokenizer, general_model)
    #예측 함수들 모임들
    NEROut = ner_predict([sentence], tokenizer, NER_model, NER_converter)
    EmoOut = emo_predict([sentence], tokenizer, Emo_model, Emo_converter)

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

def get_converters():
    path = 'C:\MyProjects\Other Projects\AI ChatBot Project\Training_Model_Weights'
    tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
    with open(path+'/Transformer_chatbot3/tokenizer.pickle', 'rb') as f:
        GC_tokenizer = pickle.load(f)
    with open(path+'/NER_RNN/index_to_tag.pickle', 'rb') as f:
        index_to_NERtag = pickle.load(f)
    with open(path+'\EmoClassification_KlueBERT\Index_to_EmotionWord.pickle', 'rb') as f:
        index_to_EmotionWord = pickle.load(f)
    labels = [label.strip() for label in open('C:\MyProjects\Other Projects\AI ChatBot Project/ner_label.txt', 'r', encoding='utf-8')]

    return (tokenizer, GC_tokenizer, index_to_NERtag, index_to_EmotionWord, labels)

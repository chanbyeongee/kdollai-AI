from AITransformer import transformer, predict
from AIKoBERT import TFBertForTokenClassification, ner_prediction
import pickle
from collections import OrderedDict

def load_general_corpus_model():
    D_MODEL = 256
    NUM_LAYERS = 2
    NUM_HEADS = 8
    DFF = 512
    DROPOUT = 0.1
    
    with open('C:/MyProjects/Other Projects/AI ChatBot Project/Training_Model_Weights/Transformer_chatbot3/tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

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

    return new_model, tokenizer

def load_NER_model():
    labels = [label.strip() for label in open('C:\MyProjects\Other Projects\AI ChatBot Project/ner_label.txt', 'r', encoding='utf-8')]
    tag_size = len(labels)

    new_model = TFBertForTokenClassification("klue/bert-base", labels=tag_size+1)

    new_model.load_weights("C:\MyProjects\Other Projects\AI ChatBot Project/Training_Model_Weights/NER_KoBERT_Keras_tuning/NER_KoBERT")

    return new_model

def To_DataStructure(input, general_model, NER_model, GCTok):
    Data = OrderedDict()
    sentence = input[0]
    name = input[1]

    GeneralAnswer = predict(sentence, GCTok, general_model)
    NEROut = ner_prediction([sentence], NER_model.tokenizer, NER_model, NER_model.index_to_tag)

    NER = {}
    for (word, tag) in NEROut[0]:
        if tag != "O":
            NER[word] = tag

    Data["Name"] = name
    Data["Input_Corpus"] = sentence
    Data["NER"] = NER
    Data["Type"] = "General"
    Data["System_Corpus"] = GeneralAnswer

    return Data

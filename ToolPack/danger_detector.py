from transformers import TFBertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os
import pickle

# tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

class detector:
    def __init__(self):
        self.bertmodel = TFBertModel.from_pretrained("klue/bert-base", from_pt=True)
        self.STFmodel = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
        self.tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
        self.danger_flag = False

        with open(os.environ['CHATBOT_ROOT']+"/resources/used_files/bad_words.txt", encoding='UTF-8') as file:
            bad_words = file.read()
            self.bad_words_list = bad_words.split(',')

        with open(os.environ['CHATBOT_ROOT']+'/resources/used_files/bad_embedding_SBERT.pickle','rb') as f:
            self.embeddings_SBERT = pickle.load(f)

        with open(os.environ['CHATBOT_ROOT']+'/resources/used_files/bad_embedding_BERT.pickle','rb') as f:
            self.embeddings_BERT = pickle.load(f)

        self.word_checked = []
        self.sentence_checked = {}

    def detect(self, sentence):
        self.check_words(sentence)
        self.check_sentence(sentence)
        self.danger_flag = False

        return self.danger_flag, self.word_checked

    def check_words(self, sentence):
        for word in self.bad_words_list:
            if word in sentence:
                self.danger_flag = True
                self.word_checked.append(word)

    def check_sentence(self, sentence):
        sample_embedding_SBERT = self.STFmodel.encode(sentence)

        sample_embedding_BERT = self.make_bertdataformat(sentence)
        bertout = self.bertmodel(input_ids=sample_embedding_BERT[0], attention_mask=sample_embedding_BERT[1], 
                                token_type_ids=sample_embedding_BERT[2])
        sample_embedding_BERT = self.mean_pooling(bertout, sample_embedding_BERT[1])

        for bad, embedding in self.embeddings_SBERT.items():
            if self.cos_sim(embedding, sample_embedding_SBERT) > 0.9:
                self.danger_flag = True
                self.sentence_checked[sentence] = bad
        
        for bad, embedding in self.embeddings_BERT.items():
            if (self.cos_sim(embedding[0], sample_embedding_BERT[0]) > 0.9) and (bad not in self.sentence_checked):
                self.danger_flag = True
                self.sentence_checked[sentence] = bad

    def cos_sim(self, A, B):
        return dot(A, B)/(norm(A)*norm(B))

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        token_embeddings = np.array(token_embeddings)
        token_embeddings = torch.tensor(token_embeddings)
        attention_mask = torch.tensor(attention_mask)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def make_bertdataformat(self, sentence):
        bert_input = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
        input_ids = np.array(bert_input['input_ids'])
        attention_mask = np.array(bert_input['attention_mask'])
        token_type_ids = np.array(bert_input['token_type_ids'])

        return (input_ids, attention_mask, token_type_ids)
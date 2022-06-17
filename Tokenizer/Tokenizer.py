from .Hanspell import Hanspell
from .KoNLPy import KoNLPy
from gensim.models import fasttext
import tensorflow as tf
import numpy as np
import json


#1. 먼저 띄어쓰기한다
#2. 띄어쓰기된 단어하나하나 교정해보면서
#3. 동시에 토크나이저를 해본다, 점수가 높은것으로 해본다.

class Tokenizer:
    def __init__(self,dim=150):
        self.spell_checker=Hanspell()
        self.tokenize = KoNLPy()
        self.dim = dim
        self.candidate_list=[]
        self.ftt_model = fasttext.load_facebook_model("..\Tokenizer\ko.bin")

    def get_dim(self):
        return self.dim , self.ftt_model.vector_size

    def auto_embedded(self, text):
        tagged = self.extract(text)
        embedded = self._embedding(tagged)

        return embedded

    def extract(self,text):
        extracted_text = self._appropriate_checker(text)

        return extracted_text

    def _embedding(self, tagged):

        embedded = []

        for (word, tag) in tagged:

            if not (tag.startswith("S")):
                try:
                    embedded.append(self.ftt_model.wv[word])
                except :
                    embedded.append(self.ftt_model.wv[self.ftt_model.most_similar(word)[0][0]])
                    print(word)

        for _ in range(self.dim-len(embedded)):
            embedded.append([0 for _ in range(self.ftt_model.vector_size)])

        embedded = np.array(embedded)
        return embedded

    def _appropriate_checker(self, text):

        raw_text_list = text.split(" ")
        temp_text_list = raw_text_list

        max_score = -99
        try:
            for (temp_idx,temp), (raw_idx,raw) in zip(enumerate(temp_text_list),enumerate(raw_text_list)):
                temp_text_list[temp_idx] = temp = self.spell_checker.check(temp)
                if temp == raw :
                    continue
                checked_text = " ".join(temp_text_list)
                tagged = self.tokenize.analyze(checked_text)

                if max_score <= self.tokenize.evaluate(tagged) :
                    raw_text_list[raw_idx] = temp

        except Exception as e :
            print(e)
            print("Target:",temp_text_list)

        checked_text = " ".join(temp_text_list)
        tagged = self.tokenize.analyze(checked_text)

        return tagged


if __name__ == "__main__":
    text="안뇽하세요? 나는 이병찬이얌."
    token = Tokenizer()

    print(token.auto_embedded(text))




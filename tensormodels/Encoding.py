import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from Utils import *

#감성대화말뭉치(최종데이터)기반

class Encoding :
    def __init__(self,file_name):
        self.load_data=pd.read_csv(file_name,encoding='cp949')
        self._indexing()
        self._del_nulls()
        self.tokenize = Tokenizer()
        self.emotion_dict = {'걱정': 0, '경멸': 1, '공포': 2, '기쁨': 3, '당혹': 4, '분노': 5,'불만':6,'불안':7,'상처':8,'슬픔':9,'중립':10}
    def _indexing(self):
        self.raw_input = self.load_data['문장']
        self.high_class = self.load_data['감정']

    def _del_nulls(self):
        self.raw_input = self.raw_input.dropna(how='any')
        self.high_class= self.high_class.dropna(how='any')

    # def embedding(self,inputs):
    #     paragraph_list = np.array([self.tokenize.auto_embedded(inputs) for inputs in inputs])
    #     self.paragraph_list += paragraph_list

    def encode(self,load):

        paragraph_list = []
        answer_list = []

        if load:
            try :
                paragraph_list = np.load("..\EmotionRecognizer\X_data.npy")
                answer_list = np.load("..\EmotionRecognizer\Y_data.npy")
            except :
                print("저장된 데이터가 없습니다!")
                print("새로운 데이터를 만듭니다.")

        if len(paragraph_list) == 0 and len(answer_list) == 0 :
            print("데이터가 손실되었거나 비어있어 새로운 데이터를 만듭니다.")

            pool = mp.Pool(10)
            print("Total Data: %d"%len(self.raw_input))
            paragraph_list = pool.map(multiprocess_func,self.raw_input)
            pool.close()
            pool.join()

            paragraph_list = np.array(paragraph_list)

            answer_list = np.asarray([self.emotion_dict[inputs.strip()] for inputs in self.high_class])

            np.save("..\EmotionRecognizer\X_data.npy",paragraph_list)
            np.save("..\EmotionRecognizer\Y_data.npy",answer_list)

        print(paragraph_list.shape)
        return paragraph_list, answer_list

    def pca_data(self,load,pca_load,components=10):
        pca_X_data = []
        y_data = []

        if pca_load:
            try:
                pca_X_data = np.load("..\EmotionRecognizer\pca_X_data.npy")
                y_data = np.load("..\EmotionRecognizer\Y_data.npy")
            except:
                print("저장된 데이터가 없습니다!")
                print("새로운 데이터를 만듭니다.")
            else:
                if len(pca_X_data) == 0 :
                    print("데이터가 손실되었거나 비어있어 새로운 데이터를 만듭니다.")

        if len(pca_X_data) == 0:
            pca_model = PCA(n_components=components)
            X_data, y_data = self.encode(load=load)


            pca_model.fit(X_data)

            print("\nRatio Info:",pca_model.explained_variance_ratio_)

            pca_X_data = np.array([pca_model.transform(data) for data in X_data])

            np.save("..\EmotionRecognizer\pca_X_data.npy", pca_X_data)

        print(pca_X_data.shape)
        return pca_X_data,y_data

    def get_dim(self):
        return self.tokenize.get_dim()

    def get_NB_classes(self):
        return len(self.emotion_dict)



if __name__ == "__main__":
    Encoder = Encoding('감성대화말뭉치.csv')
    X_data,Y_data = Encoder.encode(load=True)
    print(X_data)

import keras.utils.np_utils
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv2D, MaxPooling2D, Flatten, AveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from Encoding import Encoding


class EmotionRecognizer:
    def __init__(self,filename,_model ="CNN",hidden=128,DROPOUT=0.4,pre_load_data=True,pca_load_data=True,pca_input=0):
        self.model = None
        self.encoder = Encoding(filename)

        self.input_shape = self.encoder.get_dim()

        if pca_input>0 :
            self.input_shape = (self.input_shape[0],pca_input)

        self.NB_classes = self.encoder.get_NB_classes()
        self._setup_model(_model,hidden,DROPOUT)
        self._get_data(pre_load_data,pca_input,pca_load_data)

    def _setup_model(self,_model,hidden,DROPOUT):
        if _model == "SimpleRNN":
            self.model = Sequential()
            self.model.add(Dense(hidden, input_shape=(self.input_shape[0]*self.input_shape[1],), name='Dense_layer', activation='relu'))
            self.model.add(Dropout(DROPOUT))
            self.model.add(Dense(hidden, name='dense_layer2', activation='relu'))
            self.model.add(Dropout(DROPOUT))
            self.model.add(Dense(hidden, name='dense_layer3', activation='relu'))
            self.model.add(Dropout(DROPOUT))
            self.model.add(Dense(hidden, name='dense_layer4', activation='sigmoid'))
            self.model.add(Dropout(DROPOUT))
            self.model.add(Dense(self.NB_classes, name='dense_layer_out', activation='softmax'))
            self.model.add(Dropout(DROPOUT))
            self.model.summary()
            self.model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
        elif _model == "CNN":
            self.model = Sequential()
            self.model.add(Conv2D(128,kernel_size=(5,int(self.input_shape[1]/2)),activation="ELU",padding='same',input_shape=self.input_shape+(1,)))
            self.model.add(MaxPooling2D(2, 2))
            self.model.add(Conv2D(256, kernel_size=(3, 3), activation='tanh', padding='same'))
            self.model.add(MaxPooling2D((2, 2)))
            self.model.add(Flatten())
            self.model.add(Dense(hidden, activation="relu",))
            self.model.add(Dropout(DROPOUT))
            self.model.add(Dense(self.NB_classes, activation='softmax'))
            self.model.summary()
            self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'],)

    def _get_data(self,pre_load,is_pca,pca_load):
        if is_pca>0:
            self._pca_data(pre_load,is_pca,pca_load)
        else :
            self._encode_data(pre_load)

    def _encode_data(self,pre_load):
        X_data,Y_data = self.encoder.encode(pre_load)
        X_data= X_data.reshape(-1, self.input_shape[0], self.input_shape[1], 1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_data,Y_data,train_size=0.8, random_state=5, shuffle=True)

    def _pca_data(self,pre_load,components,pca_load):
        X_data, Y_data = self.encoder.pca_data(load=pre_load,pca_load=pca_load,components=components)
        X_data = X_data.reshape(-1, self.input_shape[0], components, 1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_data, Y_data, train_size=0.8,
                                                                                random_state=5, shuffle=True)

    def fit_and_train(self, BATCH_SIZE=32, EPOCHS=100, VERBOSE=1, VALIDATION_SPLIT = 0.2):
        self.y_train = keras.utils.np_utils.to_categorical(self.y_train,num_classes=self.NB_classes)
        self.model.fit(self.X_train, self.y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

    def get_evaluate(self):
        self.y_test = keras.utils.np_utils.to_categorical(self.y_test, num_classes=self.NB_classes)
        return self.model.evaluate(self.X_test, self.y_test)


if __name__ == "__main__":
    myTF = EmotionRecognizer("감성대화말뭉치_최종.csv", _model='CNN', pca_input=50, pre_load_data=False, pca_load_data=False)
    myTF.fit_and_train()
    test_loss, test_acc = myTF.get_evaluate()
    print('Test accuracy:',test_acc)

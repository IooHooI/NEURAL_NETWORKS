from .kerasbaseestimator import KerasBaseEstimator
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Input
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences


class KerasPreTrainedRegressor(KerasBaseEstimator):

    def __init__(self, chkpt_dir='./', maxlen=50, max_features=200, lr=0.001, b_size=128, n_epochs=6):
        super().__init__(chkpt_dir, lr, b_size, n_epochs)
        self.maxlen = maxlen
        self.max_features = max_features

    def build_the_graph(self, input_shape, output_shape):
        self.model.add(Conv1D(128, 5, activation='relu'))
        self.model.add(MaxPooling1D(5))
        self.model.add(Conv1D(128, 5, activation='relu'))
        self.model.add(MaxPooling1D(5))
        self.model.add(Conv1D(128, 5, activation='relu'))
        self.model.add(MaxPooling1D(35))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(output_shape, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))

    def __prepare_the_data(self, X):
        encoded_docs = [one_hot(' '.join(d), self.max_features) for d in X]
        padded_docs = pad_sequences(encoded_docs, maxlen=self.maxlen, padding='post')
        return padded_docs

    def fit(self, X, y=None):
        X = self.__prepare_the_data(X)

        self.model.add(Input())
        self.model.add(Embedding())

        self.build_the_graph(X.shape[1], y.reshape(len(y), 1))

        super().fit(X, y.reshape(len(y), 1))

    def predict(self, X, y=None):
        X = self.__prepare_the_data(X)
        return self.model.predict(X, batch_size=self.batch_size, verbose=1)

    def predict_proba(self, X, y=None):
        raise NotImplementedError("This method is not implemented for this algorithm")

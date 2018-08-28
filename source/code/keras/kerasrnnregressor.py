from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.optimizers import Adam
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

from .kerasbaseestimator import KerasBaseEstimator


class KerasRNNRegressor(KerasBaseEstimator):

    def __init__(self, checkpoint_dir='./', maxlen=50, max_features=200, lr=0.001, batch_size=128, n_epochs=6):
        super().__init__(checkpoint_dir, lr, batch_size, n_epochs)
        self.maxlen = maxlen
        self.max_features = max_features

    def build_the_graph(self, input_shape, output_shape):
        self.model.add(Bidirectional(LSTM(64)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(output_shape, activation='linear', kernel_initializer='normal'))
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))

    def __prepare_the_data(self, X):
        encoded_docs = [one_hot(' '.join(d), self.max_features) for d in X]
        padded_docs = pad_sequences(encoded_docs, maxlen=self.maxlen, padding='post')
        return padded_docs

    def fit(self, X, y=None):
        X = self.__prepare_the_data(X)
        self.model.add(Embedding(self.max_features, 128, input_length=self.maxlen))
        self.build_the_graph(X.shape[1], y.reshape(len(y), 1))
        super().fit(X, y.reshape(len(y), 1))

    def predict(self, X, y=None):
        X = self.__prepare_the_data(X)
        return self.model.predict(X, batch_size=self.batch_size, verbose=1)

    def predict_proba(self, X, y=None):
        raise NotImplementedError("This method is not implemented for this algorithm")

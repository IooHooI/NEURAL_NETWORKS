from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from .kerasbaseestimator import KerasBaseEstimator


class KerasRNNRegressor(KerasBaseEstimator):

    def __init__(self, checkpoint_dir='./', maxlen=50, max_features=200, lr=0.001, batch_size=128, n_epochs=6):
        super().__init__(checkpoint_dir, lr, batch_size, n_epochs)
        self.maxlen = maxlen
        self.max_features = max_features
        self.tokenizer = Tokenizer(num_words=self.max_features)

    def build_the_graph(self, input_shape, output_shape):
        self.model.add(Bidirectional(LSTM(64)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(output_shape, activation='linear', kernel_initializer='normal'))
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))

    def fit(self, X, y=None):
        self.tokenizer.fit_on_texts([' '.join(d) for d in X])
        encoded_docs = self.tokenizer.texts_to_sequences([' '.join(d) for d in X])
        padded_docs = pad_sequences(encoded_docs, maxlen=self.maxlen, padding='post')

        self.model.add(Embedding(self.max_features, 128, input_length=self.maxlen))
        self.build_the_graph(padded_docs.shape[1], y.reshape(len(y), 1).shape[1])

        super().fit(padded_docs, y.reshape(len(y), 1))

    def predict(self, X, y=None):
        encoded_docs = self.tokenizer.texts_to_sequences([' '.join(d) for d in X])
        padded_docs = pad_sequences(encoded_docs, maxlen=self.maxlen, padding='post')

        return self.model.predict(padded_docs, batch_size=self.batch_size, verbose=1)

    def predict_proba(self, X, y=None):
        raise NotImplementedError("This method is not implemented for this algorithm")

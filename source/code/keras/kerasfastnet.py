from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling1D

from sklearn.base import BaseEstimator, ClassifierMixin
from source.code.preprocessing.utils import create_ngram_set
from source.code.preprocessing.utils import add_ngram

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

import numpy as np


class KerasFastTextRegressor(BaseEstimator, ClassifierMixin):

    def __init__(
            self,
            checkpoint_dir='./',
            embedding_dims=50,
            max_features=200,
            learning_rate=0.001,
            ngram_range=2,
            batch_size=128,
            n_epochs=6
    ):
        self.model = Sequential()
        self.embedding_dims = embedding_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir

    def __build_the_graph(self, max_features, embedding_dims):
        self.model.add(Embedding(max_features, 128, input_length=embedding_dims))
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))

    def __prepare_the_data(self, X):
        encoded_docs = [one_hot(' '.join(d), self.max_features) for d in X]
        ngram_set = set()
        for input_list in encoded_docs:
            for i in range(2, self.ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)
        start_index = self.max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        max_features = np.max(list(indice_token.keys())) + 1

        encoded_docs = add_ngram(encoded_docs, token_indice, self.ngram_range)

        padded_docs = pad_sequences(encoded_docs, maxlen=self.embedding_dims, padding='post')

        return padded_docs, max_features

    def fit(self, X, y=None):
        X, max_features = self.__prepare_the_data(X)
        self.__build_the_graph(max_features, self.embedding_dims)
        self.model.fit(
            x=X,
            y=y,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            validation_split=0.3,
            callbacks=[
                ModelCheckpoint(
                    filepath=self.checkpoint_dir,
                    save_best_only=True,
                    verbose=1
                ),
                EarlyStopping(
                    patience=3,
                    verbose=1
                )
            ]
        )

    def predict(self, X, y=None):
        X, _ = self.__prepare_the_data(X)
        return self.model.predict(X, batch_size=self.batch_size, verbose=1)

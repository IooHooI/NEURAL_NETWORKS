from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling1D

from sklearn.base import BaseEstimator, ClassifierMixin

import gensim


class KerasPreTrainedRegressor(BaseEstimator, ClassifierMixin):

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
        model_path = '../../../data/dataset/ruwikiruscorpora-nobigrams_upos_skipgram_300_5_2018.vec.gz'
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)

    def fit(self, X, y=None):
        X = self.__prepare_the_data(X)

    def predict(self, X, y=None):
        X = self.__prepare_the_data(X)
        return self.model.predict(X, batch_size=self.batch_size, verbose=1)

    def predict_proba(self, X, y=None):
        raise NotImplementedError("This method is not implemented for this algorithm")

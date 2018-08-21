from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.models import Sequential
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.optimizers import Adam

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences


class KerasRNNRegressor(BaseEstimator, ClassifierMixin):

    def __init__(self, checkpoint_dir='./', maxlen=100, max_features=1000, learning_rate=0.01, batch_size=128, n_epochs=30):
        self.model = Sequential()
        self.maxlen = maxlen
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir

    def __build_the_graph(self, max_features, maxlen):
        self.model.add(Embedding(max_features, 128, input_length=maxlen))
        self.model.add(Bidirectional(LSTM(64)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))

    def fit(self, X, y=None):
        X = list(map(lambda x: one_hot(x, n=self.max_features), X))
        X = pad_sequences(X, maxlen=self.maxlen)

        self.__build_the_graph(self.max_features, self.maxlen)
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
        X = list(map(lambda x: one_hot(x, n=self.max_features), X))
        X = pad_sequences(X, maxlen=self.maxlen)

        return self.model.predict(X, batch_size=self.batch_size, verbose=1)


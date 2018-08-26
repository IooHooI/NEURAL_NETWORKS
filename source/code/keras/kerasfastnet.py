from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from sklearn.base import BaseEstimator, ClassifierMixin


class KerasFastTextRegressor(BaseEstimator, ClassifierMixin):

    def __init__(self, checkpoint_dir='./', maxlen=50, max_features=200, learning_rate=0.01, batch_size=128, n_epochs=6):
        self.model = Sequential()
        self.maxlen = maxlen
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir

    def __build_the_graph(self, max_features, maxlen):
        # TODO: implement the network
        pass

    def __prepare_the_data(self, X):
        # TODO: implement n-gramm preparation
        pass

    def fit(self, X, y=None):
        X = self.__prepare_the_data(X)
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
        X = self.__prepare_the_data(X)
        return self.model.predict(X, batch_size=self.batch_size, verbose=1)

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from sklearn.base import BaseEstimator
from abc import abstractmethod


class KerasBaseEstimator(BaseEstimator):

    def __init__(self, checkpoint_dir='./', lr=0.01, batch_size=128, n_epochs=30):
        self.checkpoint_dir = checkpoint_dir
        self.learning_rate = lr
        self.batch_size = batch_size
        self.model = Sequential()
        self.n_epochs = n_epochs

    def fit(self, X, y=None):
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
                ),
                ReduceLROnPlateau(
                    min_lr=0.0001,
                    mode='auto',
                    verbose=1
                )
            ]
        )

    @abstractmethod
    def build_the_graph(self, input_shape, output_shape):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def predict(self, X, y=None):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def predict_proba(self, X, y=None):
        raise NotImplementedError("Please Implement this method")

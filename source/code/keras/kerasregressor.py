from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.optimizers import Adam


class KerasRegressor(BaseEstimator, ClassifierMixin):

    def __init__(self, checkpoint_dir='./', learning_rate=0.01, batch_size=128, n_epochs=30):
        self.model = Sequential()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.checkpoint_dir = checkpoint_dir

    def __build_the_graph(self, input_shape, output_shape):
        self.model.add(
            Dense(input_dim=input_shape, units=output_shape, kernel_initializer="normal", activation='linear')
        )
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))

    def fit(self, X, y=None):
        self.__build_the_graph(X.shape[1], y.shape[1])
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
        return self.model.predict(X, batch_size=self.batch_size, verbose=1)


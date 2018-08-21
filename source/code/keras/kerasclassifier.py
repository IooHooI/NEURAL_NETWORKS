import numpy as np
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.optimizers import Adam


class KerasClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, checkpoint_dir='./', classification='binary', learning_rate=0.01, batch_size=128, n_epochs=30):
        self.model = Sequential()
        self.classification = classification
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.checkpoint_dir = checkpoint_dir

    def __build_for_binary_classification(self, input_shape, output_shape):
        self.model.add(Dense(input_dim=input_shape, units=output_shape, activation='sigmoid', kernel_initializer="normal"))
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])

    def __build_for_multi_classification(self, input_shape, output_shape):
        self.model.add(Dense(input_dim=input_shape, units=output_shape, activation='softmax', kernel_initializer="normal"))
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])

    def __build_the_graph(self, input_shape, output_shape):
        if self.classification == 'binary':
            self.__build_for_binary_classification(input_shape, output_shape)
        else:
            self.__build_for_multi_classification(input_shape, output_shape)

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
        y_proba = self.model.predict(X, batch_size=self.batch_size, verbose=1)
        if self.classification == 'binary':
            y_classes = np.array([[int(round(proba[0]))] for proba in y_proba])
        else:
            y_classes = np.argmax(y_proba, 1)
        return y_classes

    def predict_proba(self, X, y=None):
        return self.model.predict(x=X, batch_size=self.batch_size, verbose=1)

import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam

from .kerasbaseestimator import KerasBaseEstimator


class KerasClassifier(KerasBaseEstimator):

    def __init__(self, checkpoint_dir='./', classification='binary', lr=0.01, batch_size=128, n_epochs=30):
        super().__init__(checkpoint_dir, lr, batch_size, n_epochs)
        self.classification = classification

    def __build_for_binary_classification(self, input_shape, output_shape):
        self.model.add(Dense(input_dim=input_shape, units=output_shape, activation='sigmoid', kernel_initializer="normal"))
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])

    def __build_for_multi_classification(self, input_shape, output_shape):
        self.model.add(Dense(input_dim=input_shape, units=output_shape, activation='softmax', kernel_initializer="normal"))
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])

    def build_the_graph(self, input_shape, output_shape):
        if self.classification == 'binary':
            self.__build_for_binary_classification(input_shape, output_shape)
        else:
            self.__build_for_multi_classification(input_shape, output_shape)

    def predict(self, X, y=None):
        y_proba = self.model.predict(X, batch_size=self.batch_size, verbose=1)
        if self.classification == 'binary':
            y_classes = np.array([[int(round(proba[0]))] for proba in y_proba])
        else:
            y_classes = np.argmax(y_proba, 1)
        return y_classes

    def predict_proba(self, X, y=None):
        return self.model.predict(x=X, batch_size=self.batch_size, verbose=1)

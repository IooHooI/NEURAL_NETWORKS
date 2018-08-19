import os
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from source.code.preprocessing.utils import next_batch


class KerasClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, checkpoint_dir='./', classification='binary', learning_rate=0.01, batch_size=128, n_epochs=30):
        self.classification = classification
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.checkpoint_dir = checkpoint_dir

    def __build_the_network(self, input_shape, output_shape):

        pass

    def fit(self, X, y=None):
        pass

    def predict(self, X, y=None):
        pass

    def predict_proba(self, X, y=None):
        pass

import unittest

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from source.code.keras.kerascnnclassifier import KerasCNNClassifier
from source.code.preprocessing.utils import create_sub_folders

import logging
import sys

logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


def fit_the_network():
    train = '../../../data/dataset/hot_dog_not_hot_dog/train'
    test = '../../../data/dataset/hot_dog_not_hot_dog/test'
    create_sub_folders('../../../data/dataset/keras_model')
    classifier = KerasCNNClassifier(
        checkpoint_dir='../../../data/dataset/keras_model/model.h5',
        lr=0.001
    )
    classifier.fit(train)
    return classifier, test


def predict_case():
    classifier, X_test = fit_the_network()

    y_test, y_pred = classifier.predict(X_test)

    return y_test, y_pred


def predict_proba_case():
    classifier, X_test = fit_the_network()

    y_test, y_pred = classifier.predict_proba(X_test)

    return y_test, y_pred


class TestKerasCNNClassifier(unittest.TestCase):

    def test_keras_classification_predict(self):
        y_test, y_pred = predict_case()

        self.assertEquals(len(y_test), len(y_pred))

        logging.getLogger().info('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))

    def test_keras_classification_predict_proba(self):
        y_test, y_pred = predict_proba_case()

        self.assertEquals(len(y_test), len(y_pred))

        logging.getLogger().info('Roc-Auc: {}'.format(roc_auc_score(y_test, y_pred)))

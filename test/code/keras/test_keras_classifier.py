import unittest

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from source.code.keras.kerasclassifier import KerasClassifier
from source.code.preprocessing.dataloader import read_and_clean_thyroid_data
from source.code.preprocessing.dataloader import read_and_clean_titanic_data
from source.code.preprocessing.utils import create_sub_folders


def fit_the_network(classification, data_loader_function):
    X, y = data_loader_function()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
    create_sub_folders('../../../data/dataset/keras_model')
    classifier = KerasClassifier(
        checkpoint_dir='../../../data/dataset/keras_model/model.h5',
        classification=classification,
        n_epochs=1060,
        learning_rate=0.001
    )

    classifier.fit(X_train, y_train)
    return classifier, X_test, y_test


def predict_case(classification, data_loader_function):
    classifier, X_test, y_test = fit_the_network(classification, data_loader_function)

    y_pred = classifier.predict(X_test, y_test)

    return y_test, y_pred


def predict_proba_case(classification, data_loader_function):
    classifier, X_test, y_test = fit_the_network(classification, data_loader_function)

    y_pred = classifier.predict_proba(X_test, y_test)

    return y_test, y_pred


class TestKerasClassifier(unittest.TestCase):

    def test_keras_binary_classification_predict(self):
        y_test, y_pred = predict_case('binary', read_and_clean_titanic_data)

        self.assertEquals(len(y_test), len(y_pred))

        print('Accuracy: {}'.format(accuracy_score(y_test[:, 0].tolist(), y_pred[:, 0].tolist())))

    def test_keras_binary_classification_predict_proba(self):
        y_test, y_pred = predict_proba_case('binary', read_and_clean_titanic_data)

        self.assertEquals(len(y_test), len(y_pred))
        self.assertTrue(roc_auc_score(y_test, y_pred) > 0.5)

        print('Roc-Auc: {}'.format(roc_auc_score(y_test[:, 0].tolist(), y_pred[:, 0].tolist())))

    def test_keras_multi_classification_predict(self):
        y_test, y_pred = predict_case('multi', read_and_clean_thyroid_data)

        self.assertEquals(len(y_test), len(y_pred))

        print('Accuracy: {}'.format(accuracy_score(np.argmax(y_test, 1), y_pred)))

    def test_keras_multi_classification_predict_proba(self):
        y_test, y_pred = predict_proba_case('multi', read_and_clean_thyroid_data)

        self.assertEquals(len(y_test), len(y_pred))
        self.assertTrue(roc_auc_score(y_test, y_pred) > 0.5)

        print('Roc-Auc: {}'.format(roc_auc_score(y_test, y_pred)))

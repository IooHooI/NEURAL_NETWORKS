import unittest
from source.code.preprocessing.dataloader import read_and_clean_titanic_data
from source.code.tensorflow.tfbinaryclassifier import TfBinaryClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split


class TestTfBinaryClassifier(unittest.TestCase):

    def test_tf_binary_classifier_predict(self):
        X, y = read_and_clean_titanic_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

        classifier = TfBinaryClassifier(
            checkpoint_dir='../../../data/dataset/tf_model',
            n_epochs=60,
            learning_rate=0.02
        )
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test, y_test)

        self.assertEquals(len(y_test[:, 0].tolist()), len(y_pred))

        print('Accuracy: {}'.format(accuracy_score(y_test[:, 0].tolist(), y_pred)))
        print('Precision: {}'.format(precision_score(y_test[:, 0].tolist(), y_pred)))
        print('Recall: {}'.format(recall_score(y_test[:, 0].tolist(), y_pred)))

    def test_tf_binary_classifier_predict_proba(self):
        X, y = read_and_clean_titanic_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

        classifier = TfBinaryClassifier(
            checkpoint_dir='../../../data/dataset/tf_model',
            n_epochs=60,
            learning_rate=0.01
        )
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict_proba(X_test, y_test)

        self.assertEquals(len(y_test[:, 0].tolist()), len(y_pred))

        print('Roc-Auc: {}'.format(roc_auc_score(y_test[:, 0].tolist(), y_pred)))

import logging
import unittest
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from source.code.keras.kerasrnn import KerasRNNRegressor
from source.code.preprocessing.dataloader import read_and_clean_feedback_data
from source.code.preprocessing.utils import create_sub_folders


def fit_the_network(data_loader_function):
    X, y = data_loader_function()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    create_sub_folders('../../../data/dataset/keras_model')
    classifier = KerasRNNRegressor(
        checkpoint_dir='../../../data/dataset/keras_model/model.h5',
        n_epochs=30,
        learning_rate=0.001
    )
    classifier.fit(X_train, y_train)
    return classifier, X_test, y_test


def predict_case(data_loader_function):
    regressor, X_test, y_test = fit_the_network(data_loader_function)

    y_pred = regressor.predict(X_test, y_test)

    return y_test, y_pred


class TestKerasRegressor(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

    def test_keras_regression_predict(self):
        y_test, y_pred = predict_case(read_and_clean_feedback_data)

        self.assertEquals(len(y_test), len(y_pred))

        self.logger.info('R2-Score: {}'.format(r2_score(y_test[:, 0].tolist(), y_pred[:, 0].tolist())))

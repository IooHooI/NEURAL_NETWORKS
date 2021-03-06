import logging
import unittest
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from source.code.keras.kerasfastnetregressor import KerasFastTextRegressor
from source.code.preprocessing.dataloader import read_and_clean_feedback_data
from source.code.preprocessing.utils import create_sub_folders

import logging
import sys


logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


def fit_the_network(data_loader_function):
    X, y = data_loader_function()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    create_sub_folders('../../../data/dataset/keras_model')
    classifier = KerasFastTextRegressor(
        chkpt_dir='../../../data/dataset/keras_model/model.h5'
    )
    classifier.fit(X_train, y_train)
    return classifier, X_test, y_test


def predict_case(data_loader_function):
    regressor, X_test, y_test = fit_the_network(data_loader_function)

    y_pred = regressor.predict(X_test, y_test)

    return y_test, y_pred


class TestKerasFastNetRegressor(unittest.TestCase):

    def test_keras_fast_net_regression_predict(self):
        y_test, y_pred = predict_case(read_and_clean_feedback_data)

        self.assertEquals(len(y_test), len(y_pred))

        logging.getLogger().info('R2-Score: {}'.format(r2_score(y_test, y_pred)))

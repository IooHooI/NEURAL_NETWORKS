import unittest

from source.code.preprocessing.dataloader import read_and_clean_boston_data
from source.code.preprocessing.dataloader import read_and_clean_thyroid_data
from source.code.preprocessing.dataloader import read_and_clean_titanic_data
from source.code.preprocessing.dataloader import read_and_clean_feedback_data


class TestDataLoader(unittest.TestCase):

    def test_read_and_clean_the_data_for_titanic(self):
        X, y = read_and_clean_titanic_data()
        self.assertEquals(len(X), len(y))

    def test_read_and_clean_the_data_for_thyroid(self):
        X, y = read_and_clean_thyroid_data()
        self.assertEquals(len(X), len(y))

    def test_read_and_clean_the_data_for_boston(self):
        X, y = read_and_clean_boston_data()
        self.assertEquals(len(X), len(y))

    def test_read_and_clean_the_data_for_feedback(self):
        X, y = read_and_clean_feedback_data()
        self.assertEquals(len(X), len(y))

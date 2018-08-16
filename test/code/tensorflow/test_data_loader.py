import unittest

from source.code.preprocessing.dataloader import read_and_clean_titanic_data
from source.code.preprocessing.dataloader import read_and_clean_thyroid_data


class TestDataLoader(unittest.TestCase):

    def test_read_and_clean_the_data_for_titanic(self):
        X, y = read_and_clean_titanic_data()
        self.assertEquals(len(X), len(y))

    def test_read_and_clean_the_data_for_thyroid(self):
        X, y = read_and_clean_thyroid_data()
        self.assertEquals(len(X), len(y))

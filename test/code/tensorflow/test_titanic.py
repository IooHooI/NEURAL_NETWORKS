import unittest
from source.code.tensorflow.titanic import read_and_clean_the_data


class TestTitanic(unittest.TestCase):

    def test_read_and_clean_the_data(self):
        X, Y = read_and_clean_the_data()
        self.assertEquals(len(X), len(Y))

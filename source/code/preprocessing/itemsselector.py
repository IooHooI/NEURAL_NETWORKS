from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return data[self.columns]

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

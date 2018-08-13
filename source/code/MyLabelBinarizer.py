from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelBinarizer


class MyLabelBinarizer(TransformerMixin):

    def __init__(self, *args, **kwargs):
        self.binarizer = LabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self.binarizer.fit(x)
        return self

    def transform(self, x, y=0):
        return self.binarizer.transform(x)

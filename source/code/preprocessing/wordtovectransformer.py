import gensim
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class WordToVecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, size=100):
        self.size = size
        self.word2vec = dict()

    def fit(self, X, y=None):
        model = gensim.models.Word2Vec(X, size=self.size)
        self.word2vec.update(dict(zip(model.wv.index2word, model.wv.vectors)))
        return self

    def transform(self, X):
        return np.array(
            [np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.size)], axis=0) for words
             in X])

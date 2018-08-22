from pymystem3 import Mystem
import re


class Preprocessor:

    def __init__(self, mapping):
        self.m = Mystem()
        self.mapping = mapping

    def _filter(self, word):
        return any(letter.isalpha() for letter in word)

    def process(self, text):
        processed = re.sub('[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]', ' ', text) if text is not None else None
        processed = re.sub('[0-9]', ' ', processed) if processed is not None else None
        processed = processed.lower() if processed is not None else None
        processed = self.m.lemmatize(processed) if processed is not None else ''
        processed = list(filter(self._filter, processed)) if processed is not None else ''
        return processed

from pymystem3 import Mystem


class Preprocessor:
    def __init__(self, mapping):
        self.m = Mystem()
        self.mapping = mapping

    def process(self, text, postags=True):
        processed = self.m.analyze(text if text is not None else '')
        tagged = []
        for w in processed:
            try:
                lemma = w["analysis"][0]["lex"].lower().strip()
                pos = w["analysis"][0]["gr"].split(',')[0]
                pos = pos.split('=')[0].strip()
                pos = self.mapping.get(pos, 'X')
                tagged.append(lemma.lower() + '_' + pos)
            except KeyError:
                continue
            except IndexError:
                continue
        if not postags:
            tagged = [t.split('_')[0] for t in tagged]
        return tagged

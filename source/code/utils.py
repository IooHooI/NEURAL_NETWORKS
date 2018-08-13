from sklearn.pipeline import Pipeline
from source.code.ItemSelector import ItemSelector
from source.code.MyLabelBinarizer import MyLabelBinarizer


def generate_features_names(bin_features, cat_features, num_features):
    res = []
    if len(bin_features) > 0:
        res += bin_features
    if len(cat_features) > 0:
        for cat_feature in cat_features:
            res += list(map(lambda x: cat_feature + '_' + str(x), range(cat_features[cat_feature])))
    if len(num_features) > 0:
        res += num_features
    return res


def generate_cat_feature_counts(df, cat_features):
    return dict(zip(cat_features, list(map(lambda cat: df[cat].nunique(), cat_features))))


def generate_binarized_pipeline(column):
    return (column, Pipeline([
        ('choose', ItemSelector(column)),
        ('binarize', MyLabelBinarizer())
    ]))

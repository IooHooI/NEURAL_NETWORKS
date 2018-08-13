import pandas as pd
from source.code.utils import generate_cat_feature_counts
from source.code.utils import generate_features_names
from source.code.utils import generate_binarized_pipeline

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from source.code.ItemSelector import ItemSelector


def read_and_clean_the_data():
    titanic = pd.read_excel('../../../data/dataset/titanic3.xls')

    titanic.age.fillna(titanic.age.mean(), inplace=True)
    titanic.fare.fillna(titanic.age.mean(), inplace=True)

    titanic.sex.replace({'male': 0, 'female': 1}, inplace=True)
    titanic.embarked.replace({'S': 0, 'C': 1, 'Q': 2}, inplace=True)

    titanic = titanic[~titanic.embarked.isnull()]

    num_features = ['age', 'fare']
    cat_features = ['pclass', 'sibsp', 'parch', 'embarked']
    bin_features = ['sex']

    X = titanic[num_features + cat_features + bin_features]

    pipeline = Pipeline([
        ('union', FeatureUnion(
            [('bin', Pipeline([('choose', ItemSelector(bin_features))]))] + \
            list(map(generate_binarized_pipeline, cat_features)) + \
            [('num', Pipeline([('choose', ItemSelector(num_features))]))]
        ))
    ])

    ext_features = generate_features_names(bin_features, generate_cat_feature_counts(X, cat_features), num_features)

    X = pd.DataFrame(pipeline.fit_transform(X), columns=ext_features)

    Y = titanic.survived

    return X, Y


import pandas as pd
from source.code.ItemSelector import ItemSelector

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


def read_and_clean_the_data():
    titanic = pd.read_excel('../../../data/dataset/titanic3.xls')

    titanic.age.fillna(titanic.age.mean(), inplace=True)
    titanic.fare.fillna(titanic.fare.mean(), inplace=True)

    titanic.sex.replace({'male': 0, 'female': 1}, inplace=True)
    titanic.embarked.replace({'S': 0, 'C': 1, 'Q': 2}, inplace=True)

    titanic = titanic[~titanic.embarked.isnull()]

    num_features = ['age', 'fare']
    cat_features = ['pclass', 'embarked', 'parch', 'sibsp']
    # , 'embarked', 'parch', 'sibsp'
    bin_features = ['sex']

    X = titanic[num_features + cat_features + bin_features]

    pipeline = Pipeline([
        ('union', FeatureUnion(
            [('bin', Pipeline([('choose', ItemSelector(bin_features))]))] +
            # list(map(generate_binarized_pipeline, cat_features)) +
            [('num', Pipeline([('choose', ItemSelector(num_features))]))]
        )),
        ('poly', PolynomialFeatures(degree=2)),
        ('scale', StandardScaler())
    ])

    X = pipeline.fit_transform(X)

    y = titanic.survived.values
    y = y.reshape([len(y), 1])

    return X, y

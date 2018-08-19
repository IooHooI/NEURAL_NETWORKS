import pandas as pd
import numpy as np
from source.code.preprocessing.itemsselector import ItemSelector
from source.code.preprocessing.mylabelbinarizer import MyLabelBinarizer

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston


def read_and_clean_titanic_data():
    # TODO: write file downloader
    titanic = pd.read_excel('../../../data/dataset/titanic3.xls')

    titanic.age.fillna(titanic.age.mean(), inplace=True)
    titanic.fare.fillna(titanic.fare.mean(), inplace=True)

    titanic.sex.replace({'male': 0, 'female': 1}, inplace=True)
    titanic.embarked.replace({'S': 0, 'C': 1, 'Q': 2}, inplace=True)

    titanic = titanic[~titanic.embarked.isnull()]

    num_features = ['age', 'fare']
    cat_features = ['pclass', 'embarked', 'parch', 'sibsp']
    bin_features = ['sex']

    pipeline = Pipeline([
        ('union', FeatureUnion([
            ('bin', Pipeline(
                [
                    ('choose', ItemSelector(bin_features))
                ]
            )),
            ('num', Pipeline(
                [
                    ('choose', ItemSelector(num_features)),
                    ('scale', StandardScaler())
                ]
            ))
        ]))
    ])

    X = titanic[num_features + cat_features + bin_features]
    X = pipeline.fit_transform(X)

    y = titanic.survived.values
    y = y.reshape([len(y), 1])

    return X, y


def read_and_clean_thyroid_data():
    # TODO: write file downloader
    hypothyroid = pd.read_csv('../../../data/dataset/dataset_57_hypothyroid.csv')

    hypothyroid.sex.replace({'M': 0, 'F': 1}, inplace=True)

    hypothyroid.replace({'f': 0, 't': 1}, inplace=True)

    hypothyroid.replace({'?': 0}, inplace=True)

    hypothyroid.drop(['TBG', 'TBG_measured'], axis=1, inplace=True)

    num_features = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    cat_features = ['referral_source']
    bin_features = [
        'sex',
        'on_thyroxine',
        'query_on_thyroxine',
        'on_antithyroid_medication',
        'sick',
        'pregnant',
        'thyroid_surgery',
        'I131_treatment',
        'query_hypothyroid',
        'query_hyperthyroid',
        'lithium',
        'goitre',
        'tumor',
        'hypopituitary',
        'psych',
        'TSH_measured',
        'T3_measured',
        'TT4_measured',
        'T4U_measured',
        'FTI_measured'
    ]

    for feature in num_features:
        hypothyroid[feature] = hypothyroid[feature].astype(np.float32)
        hypothyroid[feature].fillna(hypothyroid[feature].mean(), inplace=True)

    pipeline = Pipeline([
        ('union', FeatureUnion([
            ('bin', Pipeline(
                [
                    ('choose', ItemSelector(bin_features))
                ]
            )),
            ('num', Pipeline(
                [
                    ('choose', ItemSelector(num_features)),
                    ('scale', StandardScaler())
                ]
            )),
            ('cat', Pipeline(
                [
                    ('choose', ItemSelector(cat_features)),
                    ('binarize', MyLabelBinarizer())
                ]
            ))
        ]))
    ])

    X = hypothyroid[num_features + cat_features + bin_features]
    X = pipeline.fit_transform(X)

    y = MyLabelBinarizer().fit_transform(hypothyroid.Class)

    return X, y


def read_and_clean_boston_data():
    X, y = load_boston(return_X_y=True)

    X = StandardScaler().fit_transform(X)
    y = y.reshape([len(y), 1])

    return X, y

import os
import json
import numpy as np
import pandas as pd
import requests

from sklearn.datasets import load_boston
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from source.code.preprocessing.itemsselector import ItemSelector
from source.code.preprocessing.mylabelbinarizer import MyLabelBinarizer
from source.code.preprocessing.utils import create_sub_folders


data_sources_description = '../../../data/data_sources.json'
local_path = '../../../data/dataset'


def download_data_from(from_param, to_param):
    file_name = '{}.{}'.format(from_param['name'], from_param['fmt'])
    file_path = os.path.join(to_param, file_name)
    if not os.path.exists(to_param):
        create_sub_folders(to_param)
    if not os.path.exists(file_path):
        response = requests.get(from_param['link'], stream=True)
        with open(file_path, "wb") as handle:
            for data in tqdm(response.iter_content()):
                handle.write(data)
    return file_path


def read_and_clean_titanic_data():
    data_sources = json.load(open(data_sources_description, 'r'))

    titanic = pd.read_excel(download_data_from(data_sources[0], local_path))

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
    data_sources = json.load(open(data_sources_description, 'r'))

    hypothyroid = pd.read_csv(download_data_from(data_sources[1], local_path))

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

    X = PolynomialFeatures().fit_transform(X)

    y = y.reshape([len(y), 1])

    return X, y


def read_and_clean_feedback_data():
    data_sources = json.load(open(data_sources_description, 'r'))

    feedback = pd.read_csv(download_data_from(data_sources[2], local_path))

    X = feedback[0].values
    y = feedback[1].values

    return X, y

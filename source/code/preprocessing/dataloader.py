import os
import zipfile
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
from source.code.preprocessing.preprocessor import Preprocessor


def read_and_clean_titanic_data():
    if not os.path.exists('../../../data/dataset/titanic3.xls'):
        response = requests.get('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls', stream=True)
        if not os.path.exists('../../../data/dataset'):
            create_sub_folders('../../../data/dataset')
        with open('../../../data/dataset/titanic3.xls', "wb") as handle:
            for data in tqdm(response.iter_content()):
                handle.write(data)

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
    if not os.path.exists('../../../data/dataset/dataset_57_hypothyroid.csv'):
        response = requests.get('https://www.openml.org/data/get_csv/57/dataset_57_hypothyroid.arff', stream=True)
        if not os.path.exists('../../../data/dataset'):
            create_sub_folders('../../../data/dataset')
        with open('../../../data/dataset/dataset_57_hypothyroid.csv', "wb") as handle:
            for data in tqdm(response.iter_content()):
                handle.write(data)

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

    X = PolynomialFeatures().fit_transform(X)

    y = y.reshape([len(y), 1])

    return X, y


def read_and_clean_feedback_data():
    if not os.path.exists('../../../data/dataset/feedback.csv'):
        response = requests.get(
            'https://drive.google.com/uc?authuser=0&id=1ta66bU4HtbnG5MBD-yvy9x_36lDvViB4&export=download',
            stream=True
        )
        if not os.path.exists('../../../data/dataset'):
            create_sub_folders('../../../data/dataset')
        with open('../../../data/dataset/feedback.zip', "wb") as handle:
            for data in tqdm(response.iter_content()):
                handle.write(data)
    zip_ref = zipfile.ZipFile('../../../data/dataset/feedback.zip', 'r')
    zip_ref.extractall('../../../data/dataset/')
    zip_ref.close()
    feedbacks_cleaned = []
    with open('../../../data/dataset/feedback.csv', 'r') as f:
        feedbacks = f.readlines()
        for i in range(len(feedbacks)):
            feedbacks_cleaned.append(feedbacks[i].split(',', maxsplit=4))
        for i in range(1, len(feedbacks)):
            feedbacks_cleaned[i][-1] = feedbacks_cleaned[i][-1].replace('"', '').rstrip()
            feedbacks_cleaned[i][-1] = '"' + feedbacks_cleaned[i][-1] + '"\n'
        for i in range(len(feedbacks)):
            feedbacks_cleaned[i] = ','.join(feedbacks_cleaned[i])
    with open('../../../data/dataset/feedback.csv', 'w') as f:
        f.writelines(feedbacks_cleaned)

    feedbacks = pd.read_csv(
        '../../../data/dataset/feedback.csv',
        delimiter=',',
        sep=',',
        engine='python',
        error_bad_lines=False
    )

    mapping_url = 'https://raw.githubusercontent.com/akutuzov/universal-pos-tags' \
                  '/4653e8a9154e93fe2f417c7fdb7a357b7d6ce333' \
                  '/ru-rnc.map'

    mystem2upos = {}
    r = requests.get(mapping_url, stream=True)
    for pair in r.text.split('\n'):
        pair = pair.split()
        if len(pair) > 1:
            mystem2upos[pair[0]] = pair[1]

    phrases_processor = Preprocessor(mystem2upos)

    y = feedbacks.rating.values.reshape([-1, 1])
    X = feedbacks.feedback.values
    X = list(map(lambda x: phrases_processor.process(x, postags=False), X))

    return X, y

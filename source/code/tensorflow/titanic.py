import pandas as pd


def read_and_clean_the_data():
    titanic = pd.read_excel('../../../data/dataset/titanic3.xls')

    titanic.age.fillna(titanic.age.mean(), inplace=True)
    titanic.fare.fillna(titanic.age.mean(), inplace=True)

    titanic.sex.replace({'male': 0, 'female': 1}, inplace=True)
    titanic.embarked.replace({'S': 0, 'C': 1, 'Q': 2}, inplace=True)

    titanic = titanic[~titanic.embarked.isnull()]

    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

    X = titanic[features]
    Y = titanic.survived


import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import pickle
import tensorflow as tf


class LoadAndTransform:
    def load_data(path:str) -> pd.DataFrame:
        """
        :param path: path to tranformed data from Stroke database
        :return: pd.Dataframe with raw data
        """
        raw_data = pd.read_csv(path)
        return raw_data

    def transform_data(raw_data: pd.DataFrame) -> Any:
        """
        :param raw_data: pd.Dataframe with raw from Stroke database
        :return: X_train_smote: training data oversampled with SMOTE algorithm
                 y_train_smote: training labels oversampled with SMOTE algorithm
                 y_train, y_test: test data and labels without oversampling
        """
        data,labels = raw_data.iloc[:,:7],raw_data['stroke']
        RANDOM_SEED = 40
        tf.random.set_seed(RANDOM_SEED)
        X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                            test_size=0.3,
                                                            random_state=RANDOM_SEED,
                                                            shuffle=True,
                                                            )
        oversampled = SMOTE(random_state=0)
        X_train_smote, y_train_smote = oversampled.fit_sample(X_train, y_train)
        return X_train_smote, X_test, y_train_smote, y_test

    
class LDA_Model:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.classifier_smote = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                                                           solver='svd', store_covariance=False, tol=0.0001)

    def train_and_save(self):
        model = self.classifier_smote.fit(self.x_train, self.y_train)
        filename = 'stroke_model_LDA_scikit.sav'
        pickle.dump(model, open(filename, 'wb'))


if '__main__' == __name__:
    lt = LoadAndTransform
    raw_data = lt.load_data('../API/data/stroke_data_transformed.csv')
    X_train, X_test, y_train, y_test = lt.transform_data(raw_data)
    lda = LDA_Model(X_train, X_test, y_train, y_test)
    lda.train_and_save()
    loaded_model = pickle.load(open('stroke_model_LDA_scikit.sav', 'rb'))
    y_pred = loaded_model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("f1_score:", metrics.f1_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    

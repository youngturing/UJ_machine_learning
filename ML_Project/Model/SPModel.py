import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import Any
from sklearn.model_selection import train_test_split
import pickle

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
        X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                            test_size=0.3,
                                                            random_state=0,
                                                            shuffle=True,
                                                            stratify=labels)
        oversampled = SMOTE(random_state=0)
        X_train_smote, y_train_smote = oversampled.fit_sample(X_train, y_train)
        return X_train_smote, y_train_smote, y_train, y_test

class LDA_Model:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.classifier_smote = LinearDiscriminantAnalysis()

    def train_and_save(self):
        model = self.classifier_smote.fit(self.x_train, self.x_test)
        filename = 'stroke_model.sav'
        pickle.dump(model, open(filename, 'wb'))



if '__main__' == __name__:
    lt = LoadAndTransform
    raw_data = lt.load_data('../API/data/stroke_data_transformed.csv')
    x_train, x_test, y_train, y_test = lt.transform_data(raw_data)
    lda = LDA_Model(x_train, x_test, y_train, y_test)
    lda.train_and_save()
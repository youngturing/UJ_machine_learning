import pickle
from typing import Any

import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class LoadAndTransform:
    def load_data(path: str) -> pd.DataFrame:
        """
        :param path: path to tranformed data from Stroke database
        :return: pd.Dataframe with raw data
        """
        raw_data = pd.read_csv(path)
        return raw_data

    def transform_data(raw_data: pd.DataFrame) -> Any:
        """
        :param raw_data: pd.Dataframe with raw from Stroke database
        :return: x_train_smote: training data oversampled with SMOTE algorithm
                 y_train_smote: training labels oversampled with SMOTE algorithm
                 y_train, y_test: test data and labels without oversampling
        """
        data, labels = raw_data.iloc[:, :10], raw_data['stroke']
        random_seed = 40
        tf.random.set_seed(random_seed)
        x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                            test_size=0.3,
                                                            random_state=random_seed,
                                                            shuffle=True,
                                                            )
        oversampled = SMOTE(random_state=0)
        x_train_smote, y_train_smote = oversampled.fit_sample(x_train, y_train)
        return x_train_smote, x_test, y_train_smote, y_test


class LDAModel:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.lda = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                                              solver='svd', store_covariance=False, tol=0.0001)

        self.decision_tree = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                                                    max_depth=None, max_features=None, max_leaf_nodes=None,
                                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                                    min_samples_leaf=1, min_samples_split=2,
                                                    min_weight_fraction_leaf=0.0, presort='deprecated',
                                                    random_state=2048, splitter='best')

        self.log_reg = LogisticRegression(multi_class='ovr', max_iter=200)
        self.models = [self.lda, self.decision_tree, self.log_reg]
        self.names = ['stroke_model_LDA_scikit.sav', 'stroke_model_DT_scikit.sav', 'stroke_model_LogReg_scikit.sav']

    def train_and_save(self):
        for i, j in zip(self.models, self.names):
            model = i.fit(self.x_train, self.y_train)
            pickle.dump(model, open(j, 'wb'))


def main():
    lt = LoadAndTransform
    raw_data = lt.load_data('C:/Users/miko5/Desktop/TDS/stroke_data_transformed.csv')
    x_train, x_test, y_train, y_test = lt.transform_data(raw_data)
    lda = LDAModel(x_train, x_test, y_train, y_test)
    lda.train_and_save()
    names = ['stroke_model_LDA_scikit.xlsx', 'stroke_model_DT_scikit.xlsx', 'stroke_model_LogReg_scikit.xlsx']
    for i, j in zip(lda.names, names):
        loaded_model = pickle.load(open(i, 'rb'))
        y_pred = loaded_model.predict(x_test)
        data = {'Accuracy': [metrics.accuracy_score(y_test, y_pred)],
                'f1_score': [metrics.f1_score(y_test, y_pred)],
                'Precision': [metrics.precision_score(y_test, y_pred)],
                'Recall': [metrics.recall_score(y_test, y_pred)]}
        print(data)
        df = pd.DataFrame(data)
        writer = pd.ExcelWriter(j, engine='openpyxl')
        df.to_excel(writer, sheet_name='Metrics')
        writer.save()
        writer.close()
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("f1_score:", metrics.f1_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))


if '__main__' == __name__:
    main()

import pickle

import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42


class LoadAndTransform:
    @staticmethod
    def _load_data(path: str) -> pd.DataFrame:
        """
        Parameters
        ----------
            path: path to transformed data from Stroke database
        Returns
        -------
            raw_data: pd.Dataframe with raw data
        """
        raw_data = pd.read_csv(path)
        return raw_data

    @staticmethod
    def transform_data(raw_data: pd.DataFrame, use_smote=True):
        """
        Parameters
        ----------
            use_smote: boolean parameter indicating using SMOTE technique
            raw_data: pd.Dataframe with raw from Stroke database
        Returns
        -------
            x_train_smote: training data oversampled with SMOTE algorithm
            y_train_smote: training labels oversampled with SMOTE algorithm
            y_train, y_test: test data and labels without oversampling
        """
        data, labels = raw_data.iloc[:, :5], raw_data['stroke']
        tf.random.set_seed(RANDOM_SEED)
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels,
            test_size=0.3,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
        if use_smote:
            oversampled = SMOTE(random_state=RANDOM_SEED)
            x_train_smote, y_train_smote = oversampled.fit_resample(x_train, y_train)
            return x_train_smote, x_test, y_train_smote, y_test
        else:
            return x_train, x_test, y_train, y_test


class Models:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.lda = LinearDiscriminantAnalysis(
            n_components=None, priors=None, shrinkage=None,
            solver='svd', store_covariance=False, tol=0.0001
        )

        self.decision_tree = DecisionTreeClassifier(
            ccp_alpha=0.0, class_weight=None, criterion='gini',
            max_depth=None, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            random_state=RANDOM_SEED, splitter='best'
        )

        self.qda = QuadraticDiscriminantAnalysis(
            priors=None, reg_param=0.0,
            store_covariance=False, tol=0.0001
        )

        self.log_reg = LogisticRegression(multi_class='ovr', max_iter=200)
        self.models = [self.lda, self.decision_tree, self.log_reg, self.qda]
        self.model_names = [
            'stroke_model_LDA_scikit.sav', 'stroke_model_DT_scikit.sav',
            'stroke_model_LogReg_scikit.sav', 'stroke_model_QDA_scikit.sav'
        ]

    def train_models_and_save(self):
        for model, model_name in zip(self.models, self.model_names):
            model = model.fit(self.x_train, self.y_train)
            pickle.dump(model, open(model_name, 'wb'))


def main():
    raw_data = LoadAndTransform._load_data('../data/stroke_data_transformed.csv')
    x_train, x_test, y_train, y_test = LoadAndTransform.transform_data(raw_data=raw_data, use_smote=True)
    models = Models(x_train, x_test, y_train, y_test)
    models.train_models_and_save()
    excel_filenames = 'stroke_models_metrics.xlsx'
    all_models_metrics = []
    for model_name in (models.model_names):
        loaded_model = pickle.load(open(model_name, 'rb'))
        y_pred = loaded_model.predict(x_test)
        data = {'Accuracy': [metrics.accuracy_score(y_test, y_pred)],
                'f1_score': [metrics.f1_score(y_test, y_pred)],
                'Precision': [metrics.precision_score(y_test, y_pred)],
                'Recall': [metrics.recall_score(y_test, y_pred)]}
        print(data)
        all_models_metrics.append(data)

    writer = pd.ExcelWriter(excel_filenames, engine='openpyxl')
    for model_name, models_metrics in zip(models.model_names, all_models_metrics):
        df = pd.DataFrame(models_metrics)
        df.to_excel(writer, sheet_name=f'{model_name}')
    writer.save()
    writer.close()


if '__main__' == __name__:
    main()
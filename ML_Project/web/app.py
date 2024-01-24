import warnings
from typing import Dict

import streamlit as st
import pandas as pd
import pickle

warnings.filterwarnings('ignore')

# header
st.title('Stroke prediction app')
st.header('Class labels')
st.write('Stroke prediction is the probability where 0% means there is no possibility of a stroke occurring '
         'according to the algorithm.'
         )

# sidebar
st.sidebar.header('User Input Parameters')
st.sidebar.info('Select all information below so that the system can make prediction')
age = st.sidebar.slider('Age', 10, 99, 1)
hypertension = st.sidebar.multiselect('Hypertension: ', options=['Yes', 'No'])
heart_disease = st.sidebar.multiselect('Heart disease: ', options=['Yes', 'No'])
ever_married = st.sidebar.multiselect('Ever married: ', options=['Yes', 'No'])
avg_glucose_level = st.sidebar.slider('Average glucose level:', 30.0, 350.0, 1.0)
make_predcition_button = st.sidebar.button('Load data and predict')

model_filename = '../model/stroke_model_LogReg_scikit.sav'


def get_user_input() -> pd.DataFrame:
    data = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'avg_glucose_level': avg_glucose_level,
    }
    user_input = pd.DataFrame(data)
    return user_input


def get_data(age, hypertension, heart_disease, ever_married, avg_glucose_level) -> Dict:
    """
    Returns
    -------
        features_for_pred: dictionary with categorical features for prediction.
    """

    if hypertension[0] == 'No':
        hypertension_features = 0
    else:
        hypertension_features = 1

    if heart_disease[0] == 'No':
        heart_disease_features = 0
    else:
        heart_disease_features = 1

    if ever_married[0] == 'No':
        ever_married_features = 0
    else:
        ever_married_features = 1

    features_for_prediction = {
        'age': [age],
        'hypertension': [hypertension_features],
        'heart_disease': [heart_disease_features],
        'ever_married': [ever_married_features],
        'avg_glucose_level': [avg_glucose_level],
    }
    return features_for_prediction


def show_data(data: dict, data_input: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
        data: dictionary with transformed data from user input
        data_input: input from user
    Returns
    -------
        features: pd.DataFrame with features for prediction
    """

    features = pd.DataFrame(data)
    features_for_pred_df = pd.DataFrame(features_for_pred, index=[0])
    features_for_pred_df['age'].astype(float)
    features_for_pred_df['hypertension'].astype(int)
    features_for_pred_df['heart_disease'].astype(int)
    features_for_pred_df['ever_married'].astype(int)
    features_for_pred_df['avg_glucose_level'].astype(float)
    st.subheader('User Input parameters:')
    st.dataframe(data_input, hide_index=True)
    st.subheader('Encoded User Input parameters:')
    st.dataframe(features_for_pred_df, hide_index=True)
    return features


def determine_color_and_write(proba: int) -> None:
    if proba < 0.5:
        st.write(':green[{:.1%}]'.format(proba))
    else:
        st.write(':red[{:.1%}]'.format(proba))


def make_predcition(filename: str, features_for_pred_df: pd.DataFrame) -> None:
    """
    Parameters
    ----------
        filename: path to model
        features_for_pred_df: pd.DataFrame with features for prediction
    Returns
    -------
        score: prediction score and subheader
    """
    loaded_model = pickle.load(open(filename, 'rb'))
    probas = loaded_model.predict_proba(features_for_pred_df)
    probability = probas[0][1]
    st.subheader('Stroke probability:')
    determine_color_and_write(proba=probability)


if make_predcition_button:
    try:
        user_input = get_user_input()
        features_for_pred = get_data(age, hypertension, heart_disease, ever_married, avg_glucose_level)
        features_for_pred_df = show_data(features_for_pred, user_input)
        make_predcition(model_filename, features_for_pred_df)
    except ValueError as e:
        st.error(f'There is missing data in your input information. Please enter a valid input.\n{e}')
    except Exception as e:
        st.error(f'Unexpected error occurred:\n{e}')

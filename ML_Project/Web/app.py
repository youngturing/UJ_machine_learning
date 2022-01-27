import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle
import warnings
warnings.filterwarnings('ignore')


st.write("""
# Stroke Prediction App
""")
st.subheader('Class labels and their corresponding index number:')
classes_df = pd.DataFrame({'Stroke':['No','Yes']})
st.write(classes_df)
#####################################################

st.sidebar.header('User Input Parameters')
age = st.sidebar.slider('Age', 10, 99, 1)
hypertension = st.sidebar.multiselect('Hypertension: ', options= ['Yes','No'])
heart_disease = st.sidebar.multiselect('Heart disease: ', options= ['Yes','No'])
ever_married = st.sidebar.multiselect('Ever married: ', options= ['Yes','No'])
work_type = st.sidebar.multiselect('Work type: ', options= ['Private','Self-employed','Government job','Childcare','Never worked'])
avg_glucose_level = st.sidebar.slider('Average glucose level:', 30.0, 350.0, 1.0)
smoking_status = st.sidebar.multiselect('Smoking status: ', options= ['never smoked','Unknown','formerly smoked','smokes'])
make_predcition_button = st.sidebar.button('Load data and predict')
data = {'age':age
    ,'hypertension':hypertension
    ,'heart_disease':heart_disease
    ,'ever_married':ever_married
    ,'work_type':work_type
    ,'avg_glucose_level':avg_glucose_level
    ,'smoking_status':smoking_status
        }

#####################################################

def get_data(age,hypertension,heart_disease,ever_married,
             work_type,avg_glucose_level,smoking_status):
    """
    :param hypertension_features: hypertension value from user input
    :param heart_disease_features: heart_disease value from user input
    :param ever_married_features: ever_married value from user input
    :param work_type_features: work_type value from user input
    :param smoking_status_features: smoking_status value from user input
    :return: dictionary with transformed data from user input
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

    if work_type[0] == 'Private':
        work_type_features = 0
    elif work_type[0] == 'Self-employed':
        work_type_features = 1
    elif work_type[0] == 'Government job':
        work_type_features = 2
    elif work_type[0] == 'Childcare':
        work_type_features = 3
    elif work_type[0] == 'Never worked':
        work_type_features = 4

    if smoking_status[0] == 'never smoked':
        smoking_status_features = 0
    elif smoking_status[0] == 'Unknown':
        smoking_status_features = 1
    elif smoking_status[0] == 'formerly smoked':
        smoking_status_features = 2
    elif smoking_status[0] == 'smokes':
        smoking_status_features = 3

    features_for_pred = {'age': [age]
        , 'hypertension': [hypertension_features]
        , 'heart_disease': [heart_disease_features]
        , 'ever_married': [ever_married_features]
        , 'work_type': [work_type_features]
        , 'avg_glucose_level': [avg_glucose_level]
        , 'smoking_status': [smoking_status_features]
                         }
    return features_for_pred

def show_data(data: dict):
    """
    :param data: dictionary with transformed data from user input
    :return pd.DataFrame with features for prediction
    """
    features = pd.DataFrame(data)
    features_for_pred_df = pd.DataFrame(features_for_pred, index=[0])
    features_for_pred_df['age'].astype(np.float)
    features_for_pred_df['hypertension'].astype(np.int)
    features_for_pred_df['heart_disease'].astype(np.int)
    features_for_pred_df['ever_married'].astype(np.int)
    features_for_pred_df['work_type'].astype(np.int)
    features_for_pred_df['avg_glucose_level'].astype(np.float)
    features_for_pred_df['smoking_status'].astype(np.int)
    st.subheader('User Input parameters:')
    st.write(features)
    st.subheader('Encoded User Input parameters:')
    st.write(features_for_pred_df)
    return features_for_pred_df

def make_predcition(filename: str,features_for_pred_df: pd.DataFrame):
    """
    :param filename: path to model
    :param features_for_pred_df: pd.DataFrame with features for prediction
    :return: prediction score and subheader
    """
    loaded_model = pickle.load(open(filename, 'rb'))
    score = loaded_model.predict_proba(features_for_pred_df)
    score_pred = "{:.2f}".format((score[0][1]))
    st.subheader(f'Prediction Probability: {score_pred}% ')
    return score

def prediction_score_plot(score: float):
    """
    :param score: probability of stroke
    :return: plot with probablity level
    """
    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 6}
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(3, 3.3))
    plt.gca().cla()
    xlab = ['Score']
    plt.ylim(0.0, 100.0)
    plt.bar(xlab, score, align='center', color='red',width=0.1)
    plt.ylabel(f'Predicted stroke probability')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.savefig('risk_plot.png')
    st.image("risk_plot.png")

#####################################################

if make_predcition_button:
    try:
        features_for_pred = get_data(age,hypertension,heart_disease,ever_married,
                                     work_type,avg_glucose_level,smoking_status)
        features_for_pred_df = show_data(features_for_pred)
        filename = 'C:/Users/miko5/Desktop/TDS/UJ_ML/ML_Project/Model/stroke_model_LogReg_scikit.sav'
        score = make_predcition(filename,features_for_pred_df)
        prediction_score_plot(score[0][1] * 100)
    except ValueError as e:
        st.error(f'There is missing data in your input information. Please enter a valid input.{e}')
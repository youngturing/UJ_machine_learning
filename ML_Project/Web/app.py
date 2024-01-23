import numpy as np
import streamlit as st
import pandas as pd
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
gender = st.sidebar.multiselect('Gneder', options = ['Male','Female'])
age = st.sidebar.slider('Age', 10, 99, 1)
hypertension = st.sidebar.multiselect('Hypertension: ', options= ['Yes','No'])
heart_disease = st.sidebar.multiselect('Heart disease: ', options= ['Yes','No'])
ever_married = st.sidebar.multiselect('Ever married: ', options= ['Yes','No'])
work_type = st.sidebar.multiselect('Work type: ', options= ['Private','Self-employed','Government job','Childcare','Never worked'])
residence_type = st.sidebar.multiselect('Residence type: ', options= ['Rural','Urban'])
avg_glucose_level = st.sidebar.slider('Average glucose level:', 30.0, 350.0, 1.0)
bmi = st.sidebar.slider('BMI:', 15.0, 70.0, 0.5)
smoking_status = st.sidebar.multiselect('Smoking status: ', options= ['never smoked','Unknown','formerly smoked','smokes'])
make_predcition_button = st.sidebar.button('Load data and predict')
data = {
    'gender': gender,
    'age':age,
    'hypertension':hypertension,
    'heart_disease':heart_disease,
    'ever_married':ever_married,
    'residence_type': residence_type,
    'work_type':work_type,
    'avg_glucose_level':avg_glucose_level,
    'bmi': bmi,
    'smoking_status':smoking_status,
        }

input = pd.DataFrame(data)

#####################################################

def get_data(gender, age, hypertension, heart_disease, ever_married,
             work_type, residence_type, avg_glucose_level, bmi, smoking_status):
    """
    :param hypertension_features: hypertension value from user input
    :param heart_disease_features: heart_disease value from user input
    :param ever_married_features: ever_married value from user input
    :param work_type_features: work_type value from user input
    :param smoking_status_features: smoking_status value from user input
    :return: dictionary with transformed data from user input
    """
    if gender[0] == 'Male':
        gender_features = 0
    else:
        gender_features = 1

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

    if residence_type[0] == 'Rural':
        residence_features = 0
    else:
        residence_features = 1

    if smoking_status[0] == 'never smoked':
        smoking_status_features = 0
    elif smoking_status[0] == 'Unknown':
        smoking_status_features = 1
    elif smoking_status[0] == 'formerly smoked':
        smoking_status_features = 2
    elif smoking_status[0] == 'smokes':
        smoking_status_features = 3

    features_for_pred = {
        'gender': [gender_features],
        'age': [age],
        'hypertension': [hypertension_features],
        'heart_disease': [heart_disease_features],
        'ever_married': [ever_married_features],
        'work_type': [work_type_features],
        'residence_type': [residence_features],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status_features],
    }
    return features_for_pred

def show_data(data: dict, data_input: pd.DataFrame):
    """
    :param data: dictionary with transformed data from user input
    :param data_input: input from user
    :return pd.DataFrame with features for prediction
    """
    features = pd.DataFrame(data)
    features_for_pred_df = pd.DataFrame(features_for_pred, index=[0])
    features_for_pred_df['gender'].astype(np.int)
    features_for_pred_df['age'].astype(np.float)
    features_for_pred_df['hypertension'].astype(np.int)
    features_for_pred_df['heart_disease'].astype(np.int)
    features_for_pred_df['ever_married'].astype(np.int)
    features_for_pred_df['work_type'].astype(np.int)
    features_for_pred_df['residence_type'].astype(np.int)
    features_for_pred_df['avg_glucose_level'].astype(np.float)
    features_for_pred_df['bmi'].astype(np.float)
    features_for_pred_df['smoking_status'].astype(np.int)
    st.subheader('User Input parameters:')
    st.write(data_input)
    st.subheader('Encoded User Input parameters:')
    st.write(features_for_pred_df)
    return features

def make_predcition(filename: str,features_for_pred_df: pd.DataFrame):
    """
    :param filename: path to model
    :param features_for_pred_df: pd.DataFrame with features for prediction
    :return: prediction score and subheader
    """
    loaded_model = pickle.load(open(filename, 'rb'))
    score = loaded_model.predict(features_for_pred_df)
    print(score)
    if score[0] == 0:
        st.subheader(f'Prediction Probability: low probality')
        st.write(score)
    else:
        st.subheader(f'Prediction Probability: plausible')
        st.write(score)
    return score

#####################################################

if make_predcition_button:
    try:
        features_for_pred = get_data(gender, age, hypertension, heart_disease, ever_married,
                                     work_type, residence_type, avg_glucose_level, bmi, smoking_status)
        features_for_pred_df = show_data(features_for_pred,input)
        filename = 'C:/Users/miko5/Desktop/TDS/uj-ml/ML_Project/Model/stroke_model_LogReg_scikit.sav'
        score = make_predcition(filename,features_for_pred_df)
    except ValueError as e:
        st.error(f'There is missing data in your input information. Please enter a valid input.{e}')

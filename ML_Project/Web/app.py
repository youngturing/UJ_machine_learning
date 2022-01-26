import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

st.write("""
# Simple Stroke Probability Prediction App
""")
st.subheader('Class labels and their corresponding index number:')
classes_df = pd.DataFrame({'Stroke':['No','Yes']})
st.write(classes_df)
st.subheader('User Input parameters:')

#####################################################

st.sidebar.header('User Input Parameters')
age = st.sidebar.slider('Age', 10, 99, 1)
hypertension = st.sidebar.multiselect('Hypertension: ', options= ['Yes','No'])
heart_disease = st.sidebar.multiselect('Heart disease: ', options= ['Yes','No'])
ever_married = st.sidebar.multiselect('Ever married: ', options= ['Yes','No'])
work_type = st.sidebar.multiselect('Work type: ', options= ['Private','Self-employed','Government job','Childcare','Never worked'])
residence_type = st.sidebar.multiselect('Residence type: ', options= ['Rural','Urban'])
avg_glucose_level = st.sidebar.slider('Average glucose level:', 30.0, 350.0, 1.0)
smoking_status = st.sidebar.multiselect('Smoking status: ', options= ['Never smoked','Unknown','Formerly smoked','Smokes'])
stroke = st.sidebar.multiselect('Stroke: ', options= ['Yes','No'])
make_predcition_button = st.sidebar.button('Load data')
data = {'age':age
    ,'hypertension':hypertension
    ,'heart_disease':heart_disease
    ,'ever_married':ever_married
    ,'work_type':work_type
    ,'residence_type':residence_type
    ,'avg_glucose_level':avg_glucose_level
    ,'smoking_status':smoking_status
    ,'stroke':stroke
        }
if make_predcition_button:
    try:
        features = pd.DataFrame(data)
        st.write(features)
    except ValueError:
        st.error('There is missing data in your input information. Please enter a valid input.')


st.subheader('Prediction Probability:')
score = 60
def anim_function(score,i=int):
    plt.gca().cla()
    xlab = ['Score']
    plt.ylim(0.0,100.0)
    plt.bar(xlab,score,align='center')
    plt.ylabel(f'Predicted stroke probability')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 6    }

plt.rc('font', **font)
fig, ax = plt.subplots(figsize=(1.09,1.34))
animation = FuncAnimation(fig, anim_function(score), interval=1000)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
st.pyplot(fig)
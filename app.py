import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("clf_random_forest_heart_disease.pkl", "rb"))

st.title("💓 Heart Disease Prediction (Pickle Model)")

st.sidebar.header("Input Your Health Details")


def user_input_features():
    age = st.sidebar.slider('Age', 20, 80, 50)
    sex = st.sidebar.selectbox('Sex (1 = Male, 0 = Female)', [1, 0])
    cp = st.sidebar.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting Blood Pressure', 80, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 100, 400, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = True; 0 = False)', [1, 0])
    restecg = st.sidebar.selectbox('Resting ECG Results (0-2)', [0, 1, 2])
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 60, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (1 = Yes; 0 = No)', [1, 0])
    oldpeak = st.sidebar.slider('ST depression induced by exercise', 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox('Slope of peak exercise ST segment (0-2)', [0, 1, 2])
    ca = st.sidebar.selectbox('Number of major vessels (0-3)', [0, 1, 2, 3])
    thal = st.sidebar.selectbox('Thal (1 = Normal; 2 = Fixed Defect; 3 = Reversible Defect)', [1, 2, 3])


    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    return pd.DataFrame(data, index=[0])



input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('User Input Features')
st.write(input_df)

st.subheader('Prediction')
heart_disease = np.array(['No Heart Disease', 'Heart Disease'])
st.write(heart_disease[prediction][0])

st.subheader('Prediction Probability')
st.write(prediction_proba)



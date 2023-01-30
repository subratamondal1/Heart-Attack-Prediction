from datetime import time
import numpy as np
import pandas as pd
import streamlit as st
import sklearn.linear_model as linear_model
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import pickle

# import minmax_scaler
minmax_scaler = pickle.load(open("minmax_scaler.pkl", "rb"))

# import logistic regression model
model_lr = pickle.load(open("model_lr.pkl", "rb"))

# title
st.title("Heart Attack Predictor")
st.caption(
    "This app predicts whether the patient has `Risk of Heart Attack or Not`.")

# sidebar
st.sidebar.header("Input Parameters")

target_column = "target"
categorical_columns = ["gender", "fasting_blood_sugar", "exercise_induced_chest_pain", "resting_ecg",
                       "slope_of_peak_ST_segment", "chest_pain_type", "thalassemia",
                       "fluoroscopy_colored_major_vessels"]
numerical_columns = ["ST_depression", "age",
                     "resting_bp", "max_heart_rate", "cholesterol"]

# user input


# gender M:1, F:0
gender = st.sidebar.radio(
    "**Gender**", options=("Male", "Female"), horizontal=True)

# fasting_blood_sugar Yes:1 No:0
fasting_blood_sugar = st.sidebar.radio(
    "**Fasting Blood Sugar** `> 120 mg/dl`", options=("Yes", "No"), horizontal=True)

# exercise_induced_chest_pain Yes:1 No:0
exercise_induced_chest_pain = st.sidebar.radio(
    "**Exercise Induced Chest Pain**", options=("Yes", "No"), horizontal=True)

# resting_ecg Normal:0 Abnormal:1 Hypertrophy:2
resting_ecg = st.sidebar.radio(
    "**Resting ECG**", options=("Normal", "Abnormal", "Hypertrophy"), horizontal=True)

# slope_of_peak_ST_segment Unsloping:0 Flat:1 Downsloping:2
slope_of_peak_ST_segment = st.sidebar.radio("**Slop of peak ST Segment**", options=(
    "Unsloping", "Flat", "Downsloping"), horizontal=True)

# chest_pain_type Typical:0 Atypical:1 Non-Anginal:2 Asymptomatic:3
chest_pain_type = st.sidebar.radio("**Chest Pain Type**", options=(
    "Typical", "Atypical", "Non-Anginal", "Asymptomatic"), horizontal=True)

# thalassemia Null:0 Fixed Defect:1 Normal:2 Reversible Defect:3
thalassemia = st.sidebar.radio(
    "**Thalassemia**", options=("Null", "Fixed Defect", "Normal", "Reversible Defect"), horizontal=True)

# fluoroscopy_colored_major_vessels
fluoroscopy_colored_major_vessels = st.sidebar.radio(
    "**Fluoroscopy colored major vessels**", options=(0, 1, 2, 3, 4), horizontal=True)

# ST_depression
ST_depression = st.sidebar.slider(
    "**ST depression**", min_value=0.0, max_value=10.0, value=4.2)

# age
age = st.sidebar.slider("Age", 20, 110, 50)

# resting_bp
resting_bp = st.sidebar.slider(
    "**Resting Blood Pressure** `in mm Hg`", min_value=50.0, max_value=300.0, value=160.0)

# max_heart_rate
max_heart_rate = st.sidebar.slider(
    "**Max Heart Rate Achieved**", min_value=50.0, max_value=300.0, value=95.0)

# cholesterol
cholesterol = st.sidebar.slider(
    "**Cholesterol** `in mg/dl`", min_value=100.0, max_value=800.0, value=249.0)

data = {
    "gender": gender,
    "fasting_blood_sugar": fasting_blood_sugar,
    "exercise_induced_chest_pain": exercise_induced_chest_pain,
    "resting_ecg": resting_ecg,
    "slope_of_peak_ST_segment": slope_of_peak_ST_segment,
    "chest_pain_type": chest_pain_type,
    "thalassemia": thalassemia,
    "fluoroscopy_colored_major_vessels": float(fluoroscopy_colored_major_vessels),
    "ST_depression": float(ST_depression),
    "age": float(age),
    "resting_bp": float(resting_bp),
    "max_heart_rate": float(max_heart_rate),
    "cholesterol": float(cholesterol)
}

# transposed version of data
data_display = pd.DataFrame(data, index=[0]).T

# naming the column
data_display.columns = ["Values"]

# container widget for to side by side widgets
col1, col2 = st.columns(2)

with col1:
    st.write(data_display)


with col2:
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")

    # predict
    if st.button("PREDICT"):
        # Mapping the input variables
        # gender M:1, F:0
        if gender == "Male":
            gender = 1
        else:
            gender = 0

        # fasting_blood_sugar Yes:1 No:0
        if fasting_blood_sugar == "Yes":
            fasting_blood_sugar = 1
        else:
            fasting_blood_sugar = 0

        # exercise_induced_chest_pain Yes:1 No:0
        if exercise_induced_chest_pain == "Yes":
            exercise_induced_chest_pain = 1
        else:
            exercise_induced_chest_pain = 0

        # resting_ecg Normal:0 Abnormal:1 Hypertrophy:2
        if resting_ecg == "Normal":
            resting_ecg = 0
        elif resting_ecg == "Abnormal":
            resting_ecg = 1
        else:
            resting_ecg = 2

        # slope_of_peak_ST_segment Unsloping:0 Flat:1 Downsloping:2
        if slope_of_peak_ST_segment == "Unsloping":
            slope_of_peak_ST_segment = 0
        elif slope_of_peak_ST_segment == "Flat":
            slope_of_peak_ST_segment = 1
        else:
            slope_of_peak_ST_segment = 2

        # chest_pain_type Typical:0 Atypical:1 Non-Anginal:2 Asymptomatic:3
        if chest_pain_type == "Typical":
            chest_pain_type = 0
        elif chest_pain_type == "Atypical":
            chest_pain_type = 1
        elif chest_pain_type == "Non-Anginal":
            chest_pain_type = 2
        else:
            chest_pain_type = 3

        # thalassemia Null:0 Fixed Defect:1 Normal:2 Reversible Defect:3
        if thalassemia == "Null":
            thalassemia = 0
        elif thalassemia == "Fixed Defect":
            thalassemia = 1
        elif thalassemia == "Normal":
            thalassemia = 2
        else:
            thalassemia = 3

        # input  dataframe
        input_processed_data = {
            "gender": gender,
            "fasting_blood_sugar": fasting_blood_sugar,
            "exercise_induced_chest_pain": exercise_induced_chest_pain,
            "resting_ecg": resting_ecg,
            "slope_of_peak_ST_segment": slope_of_peak_ST_segment,
            "chest_pain_type": chest_pain_type,
            "thalassemia": thalassemia,
            "fluoroscopy_colored_major_vessels": float(fluoroscopy_colored_major_vessels),
            "ST_depression": float(ST_depression),
            "age": float(age),
            "resting_bp": float(resting_bp),
            "max_heart_rate": float(max_heart_rate),
            "cholesterol": float(cholesterol)
        }

        X = pd.DataFrame(input_processed_data, index=[0])

        # input categorical columns
        input_cat_col = [gender, fasting_blood_sugar, exercise_induced_chest_pain, resting_ecg,
                         slope_of_peak_ST_segment, chest_pain_type, thalassemia,
                         fluoroscopy_colored_major_vessels]

        X_num = X[numerical_columns].to_numpy()
        robust_scaler = preprocessing.RobustScaler()
        scaled_X_num = robust_scaler.fit_transform(X_num.reshape(-1, 1))
        X[numerical_columns] = scaled_X_num.T

        result = model_lr.predict(X.to_numpy().reshape(1, -1))

        if result == 1:
            st.header("Risk")
        else:
            st.header("No Risk")

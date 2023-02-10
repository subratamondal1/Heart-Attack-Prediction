from datetime import time
import numpy as np
import pandas as pd
import streamlit as st
import sklearn.linear_model as linear_model
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import pickle


# import logistic regression model
model_lr = pickle.load(open("model_lr.pkl", "rb"))
model_best = pickle.load(open("model_with_best_params.pkl", "rb"))
pipe = pickle.load(open("pipe.pkl", "rb"))

# import data
cleaned_data = pd.read_csv("cleaned_data.csv")
final_data = pd.read_csv("final_data_pre_scaled.csv")
final_data = final_data.rename({
    "age": "age",
    "sex": "gender",
    "cp": "chest_pain_type",
    "trtbps": "resting_bp",
    "chol": "cholesterol",
    "fbs": "fasting_blood_sugar",
    "restecg": "resting_ecg",
    "thalach": "max_heart_rate",
    "exang": "exercise_induced_chest_pain",
    "old_peak": "ST_depression",
    "slp": "slope_of_peak_ST_segment",
    "caa": "fluoroscopy_colored_major_vessels",
    "thal": "thalassemia",
    "target": "target"
}, axis=1)

# title
st.title("Heart Attack Predictor")

# caption
st.caption(
    "This app predicts whether the patient has the `Risk of Heart Attack or Not`.")

# sidebar
st.sidebar.header("Input Parameters")

# target column
target_column = "target"

# categorical columns
categorical_columns = ["gender", "fasting_blood_sugar", "exercise_induced_chest_pain", "resting_ecg",
                       "slope_of_peak_ST_segment", "chest_pain_type", "thalassemia",
                       "fluoroscopy_colored_major_vessels"]

# numerical columns
numerical_columns = ["ST_depression", "age",
                     "resting_bp", "max_heart_rate", "cholesterol"]

# user input starts from here

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
    "**Thalassemia**", options=("Fixed Defect", "Normal", "Reversible Defect"), horizontal=True)

# fluoroscopy_colored_major_vessels
fluoroscopy_colored_major_vessels = st.sidebar.radio(
    "**Fluoroscopy colored major vessels**", options=(0, 1, 2, 3, 4), horizontal=True)

# ST_depression min value
ST_depression_min = float(cleaned_data["ST_depression"].min())
# ST_depression max value
ST_depression_max = float(cleaned_data["ST_depression"].max())
# ST_depression avg value
ST_depression_avg = (ST_depression_min+ST_depression_max)/2
# ST_depression
ST_depression = st.sidebar.slider(
    "**ST depression**", min_value=ST_depression_min, max_value=ST_depression_max, value=ST_depression_avg)

# age min value
age_min = int(cleaned_data["age"].min())
# age max value
age_max = int(cleaned_data["age"].max())
# age avg value
age_avg = int((age_min+age_max)/2)
# age
age = st.sidebar.slider(
    "Age", min_value=age_min, max_value=age_max, value=age_avg)

# resting_bp min value
resting_bp_min = float(cleaned_data["resting_bp"].min())
# resting_bp max value
resting_bp_max = float(cleaned_data["resting_bp"].max())
# resting_bp avg value
resting_bp_avg = (resting_bp_min + resting_bp_max)/2
# resting_bp
resting_bp = st.sidebar.slider(
    "**Resting Blood Pressure** `in mm Hg`", min_value=resting_bp_min, max_value=resting_bp_max, value=resting_bp_avg)

# max_heart_rate min value
max_heart_rate_min = float(cleaned_data["max_heart_rate"].min())
# max_heart_rate max value
max_heart_rate_max = float(cleaned_data["max_heart_rate"].max())
# max_heart_rate avg value
max_heart_rate_avg = (max_heart_rate_min + max_heart_rate_max)/2
# max_heart_rate
max_heart_rate = st.sidebar.slider(
    "**Max Heart Rate Achieved**", min_value=max_heart_rate_min, max_value=max_heart_rate_max, value=max_heart_rate_avg)

# cholesterol min value
cholesterol_min = float(cleaned_data["cholesterol"].min())
# cholesterol max value
cholesterol_max = float(cleaned_data["cholesterol"].max())
# cholesterol avg value
cholesterol_avg = (cholesterol_min+cholesterol_max)/2
# cholesterol
cholesterol = st.sidebar.slider(
    "**Cholesterol** `in mg/dl`", min_value=cholesterol_min, max_value=cholesterol_max, value=cholesterol_avg)

# data dictionary
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
    "age": int(age),
    "resting_bp": float(resting_bp),
    "max_heart_rate": float(max_heart_rate),
    "cholesterol": float(cholesterol)
}

numerical_features = ['age', 'max_heart_rate', 'resting_bp', 'ST_depression']
categorical_features = ['gender', 'chest_pain_type', 'exercise_induced_chest_pain', 'slope_of_peak_ST_segment',
                        'fluoroscopy_colored_major_vessels', 'thalassemia']

# transposed version of data
data_df = pd.DataFrame(data, index=[0])
data_display = data_df.T

# naming the column
data_display.columns = ["Values"]

# container widget for to side by side widgets
col1, col2 = st.columns(2)

with col1:
    st.write(data_display)

with col2:

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

        user_input = np.array([age, gender,	chest_pain_type, resting_bp, max_heart_rate,	exercise_induced_chest_pain,
                               ST_depression, slope_of_peak_ST_segment, fluoroscopy_colored_major_vessels, thalassemia]).reshape(1, 10)

        result = pipe.predict(user_input)

        if result == 1:
            st.header("Risk")
            st.balloons()
        else:
            st.header("No Risk")
            st.balloons()

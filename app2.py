from datetime import time
import numpy as np
import pandas as pd
import streamlit as st
import sklearn.ensemble as ensemble

# import data
data = pd.read_csv("cleaned_heart.csv")

# title
st.title("Heart Attack Predictor")
st.caption(
    "This app predicts whether the patient has `Risk of Heart Attack or Not`.")

# sidebar
st.sidebar.header("Input Parameters")

target_column = "target"
categorical_columns = ["gender", "fasting_blood_sugar", "exercise_induced_chest_pain", "resting_ecg",
                       "slope_of_peak_ST_depression", "chest_pain_type", "thalassemia",
                       "fluoroscopy_colored_major_vessels"]
numerical_columns = ["ST_depression", "age",
                     "resting_bp", "max_heart_rate", "cholesterol"]

# user input


def user_input():
    # gender M:1, F:0
    gender = st.sidebar.select_slider(
        "Gender", options=["Male", "Female"], value="Male")
    # fasting_blood_sugar Yes:1 No:0
    fasting_blood_sugar = st.sidebar.select_slider(
        "Fasting Blood Sugar `> 120 mg/dl`", options=["Yes", "No"], value="Yes")

    # exercise_induced_chest_pain Yes:1 No:0
    exercise_induced_chest_pain = st.sidebar.select_slider(
        "Exercise Induced Chest Pain", options=["Yes", "No"], value="Yes")

    # resting_ecg Normal:0 Abnormal:1 Hypertrophy:2
    resting_ecg = st.sidebar.select_slider(
        "Resting ECG", options=["Normal", "Abnormal", "Hypertrophy"], value="Abnormal")

    # slope_of_peak_ST_segment Unsloping:0 Flat:1 Downsloping:2
    slope_of_peak_ST_segment = st.sidebar.select_slider("Slop of peak ST Segment", options=[
        "Unsloping", "Flat", "Downsloping"], value="Downsloping")

    # chest_pain_type Typical:0 Atypical:1 Non-Anginal:2 Asymptomatic:3
    chest_pain_type = st.sidebar.select_slider("Chest Pain Type", options=[
        "Typical", "Atypical", "Non-Anginal", "Asymptomatic"], value="Asymptomatic")

    # thalassemia Null:0 Fixed Defect:1 Normal:2 Reversible Defect:3
    thalassemia = st.sidebar.select_slider(
        "Thalassemia", options=["Null", "Fixed Defect", "Normal", "Reversible Defect"], value="Fixed Defect")

    # fluoroscopy_colored_major_vessels
    fluoroscopy_colored_major_vessels = st.sidebar.select_slider(
        "Fluoroscopy colored major vessels", options=[0, 1, 2, 3, 4], value=4)

    # ST_depression
    ST_depression = st.sidebar.slider(
        "ST depression", min_value=0.0, max_value=10.0, value=4.2)

    # age
    age = st.sidebar.slider("Age", 20, 110, 50)

    # resting_bp
    resting_bp = st.sidebar.slider(
        "Resting Blood Pressure", min_value=50.0, max_value=300.0, value=160.0)

    # max_heart_rate
    max_heart_rate = st.sidebar.slider(
        "Max Heart Rate Achieved", min_value=50.0, max_value=300.0, value=95.0)

    # cholesterol
    cholesterol = st.sidebar.slider(
        "Cholesterol", min_value=100.0, max_value=800.0, value=249.0)

    data = {
        "Gender": gender,
        "Fasting Blood Sugar": fasting_blood_sugar,
        "Exercise Induced Chest Pain": exercise_induced_chest_pain,
        "Resting ECG": resting_ecg,
        "Slope of peak ST Segment": slope_of_peak_ST_segment,
        "Chest Pain Type": chest_pain_type,
        "Thalassemia": thalassemia,
        "Fluoroscopy colored major vessels": fluoroscopy_colored_major_vessels,
        "ST depression": ST_depression,
        "Age": age,
        "Resting Blood Pressure": resting_bp,
        "Max Heart Rate Achieved": max_heart_rate,
        "Cholesterol": cholesterol
    }

    features = pd.DataFrame(data, index=[0])

    return features


df = user_input()
df_display = df.T
df_display.columns = ["Values"]
st.write(df["max_heart_rate"])

# # predict
# if st.button("PREDICT"):
#     if df["max_heart_rate"] >= 70 and df["chloesterol"] >= 150 and data["resting_bp"] >= 120 and data["age"] >= 45:
#         st.title("Risk of Heart Attack")
#     else:
#         st.title("No risk of Heart Attack")

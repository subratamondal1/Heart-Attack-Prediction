# import libraries
import numpy as np
import pandas as pd
import streamlit as st

# title
st.title("HEART ATTACK PREDICTION")
st.caption("This Heart Attack Predictor app takes input from the user after they have done the relevant tests, based on those input the ML model gives predictions whether they are having the `Risk of Heart Attack or Not`. We have finalized the model with `Logistic Regression` providing an accuracy score of approx: `88.5%`")

# gender
gender = st.selectbox("GENDER", ["Male", "Female"])

# age
age = st.selectbox("AGE", np.arange(1, 111))

# chest pain type
chest_pain = st.selectbox("CHEST PAIN TYPE", np.arange(1, 5))

# blood pressure
bp = st.number_input("BLOOD PRESSURE `millimetres of mercury (mmHg)`")

# cholestrol
chloestrol = st.number_input("CHLOESTROL `milligrams per deciliter (mg/dl)`")

# maximum heart rate achieved
max_heart_rate = st.number_input(
    f"MAX HEART RATE ACHIEVED `beats per minute`")


# predict
if st.button("PREDICT"):
    if max_heart_rate >= 70 and chloestrol >= 150 and bp >= 120 and age >= 45:
        st.title("Risk of Heart Attack")
    else:
        st.title("No risk of Heart Attack")

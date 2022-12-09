# import libraries
import numpy as np
import pandas as pd
import streamlit as st

# title
st.title("HEART ATTACK PREDICTION")

# gender
st.selectbox("GENDER", ["Male", "Female"])

# age
st.selectbox("AGE", np.arange(1, 111))

# chest pain type
st.selectbox("CHEST PAIN TYPE", np.arange(1, 5))

# blood pressure
st.number_input("BLOOD PRESSURE")

# cholestrol
st.number_input("CHLOESTROL")

# maximum heart rate achieved
st.number_input("MAX HEART RATE ACHIEVED")

# predict
st.button("PREDICT")

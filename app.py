import streamlit as st
import joblib
import numpy as np

model = joblib.load("microbiome_model.pkl")

st.title("AI Microbiome Therapeutic Designer")

age = st.slider("Age", 10, 80)
diet = st.selectbox("Diet", ["Balanced","Fast Food","Vegetarian"])
sleep = st.slider("Sleep Hours", 4,10)

if st.button("Generate Prediction"):
    data = np.random.rand(1,45)
    pred = model.predict(data)
    st.success(f"Disease Risk: {pred[0]}")

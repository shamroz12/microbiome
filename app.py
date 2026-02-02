import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("microbiome_model_final.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="AI Microbiome Predictor", layout="centered")

st.title("AI-Driven Microbiome Stability Predictor")
st.write("This tool simulates microbiome pattern classification using AI.")

st.sidebar.header("User Input")

# Number of features must match training features
NUM_FEATURES = model.n_features_in_

st.write(f"Adjust microbial abundance sliders (Features: {NUM_FEATURES})")

# Create sliders dynamically
inputs = []
for i in range(NUM_FEATURES):
    val = st.slider(f"Feature {i+1}", 0.0, 1.0, 0.5)
    inputs.append(val)

# Predict Button
if st.button("Predict Microbiome Pattern"):
    X_input = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Pattern Group: A")
    else:
        st.warning("Pattern Group: B")

    st.write(f"Confidence Score: {prob:.2f}")

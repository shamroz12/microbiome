import streamlit as st
import numpy as np
import joblib
import os

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="AI Microbiome Therapeutic Designer",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 AI Microbiome Therapeutic Designer")
st.write("Personalized Preventive Healthcare using AI & Microbiome Simulation")

# -------------------------
# Load Model Safely
# -------------------------
model_path = "microbiome_model.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(model_path):
    st.error("Model file not found. Please upload microbiome_model.pkl")
    st.stop()

if not os.path.exists(scaler_path):
    st.error("Scaler file not found. Please upload scaler.pkl")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# -------------------------
# User Inputs
# -------------------------
st.subheader("Patient Profile")

age = st.slider("Age", 5, 80, 25)
sleep = st.slider("Sleep Hours", 3, 10, 7)
activity = st.slider("Physical Activity Level", 1, 10, 5)

diet = st.selectbox(
    "Diet Type",
    ["Balanced", "Vegetarian", "High Protein", "Fast Food"]
)

stress = st.slider("Stress Level", 1, 10, 4)

# -------------------------
# Convert Inputs to Features
# -------------------------
diet_map = {
    "Balanced": 0,
    "Vegetarian": 1,
    "High Protein": 2,
    "Fast Food": 3
}

input_features = np.array([
    age,
    sleep,
    activity,
    diet_map[diet],
    stress
])

# Expand to match PCA dimension (45 features)
# Fill remaining features with random microbiome simulation
simulated_microbiome = np.random.rand(40)

final_input = np.concatenate((input_features, simulated_microbiome))
final_input = final_input.reshape(1, -1)

# Scale
scaled_input = scaler.transform(final_input)

# -------------------------
# Prediction
# -------------------------
if st.button("Generate AI Prediction"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Higher Dysbiosis Risk Detected ({probability*100:.1f}%)")
    else:
        st.success(f"✅ Healthy Microbiome Trend ({(1-probability)*100:.1f}%)")

    # Visualization
    st.subheader("Microbiome Stability Score")
    st.progress(int((1 - probability) * 100))

# -------------------------
# Footer
# -------------------------
st.write("---")
st.caption("Research Prototype — Not for Clinical Use")

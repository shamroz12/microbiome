import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# LOAD MODEL
# ---------------------------
model = joblib.load("microbiome_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Gut Health AI", layout="wide")

st.title("🧬 Gut Microbiome Health Analyzer")
st.write("""
This tool predicts **gut health stability** using microbiome patterns.  
You will provide simple lifestyle & biological inputs — no technical data needed.
""")

# ---------------------------
# USER INPUT SECTION
# ---------------------------
st.header("🧑‍⚕️ Personal & Lifestyle Inputs")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 10, 80, 30)
    bmi = st.slider("BMI (Body Mass Index)", 15.0, 40.0, 22.0)
    sleep = st.slider("Sleep Hours / Day", 3, 10, 7)

with col2:
    diet = st.selectbox(
        "Diet Type",
        ["Balanced", "High Sugar", "High Fat", "Vegetarian", "High Fiber"]
    )
    exercise = st.slider("Exercise (Hours/Week)", 0, 10, 3)
    antibiotics = st.selectbox("Recent Antibiotic Use?", ["No", "Yes"])

st.header("🥗 Food Habit Indicators")

fiber = st.slider("Fiber Intake (0=Low,1=High)", 0.0, 1.0, 0.5)
sugar = st.slider("Sugar Intake (0=Low,1=High)", 0.0, 1.0, 0.5)
fermented = st.slider("Fermented Food Intake", 0.0, 1.0, 0.5)

# ---------------------------
# ENCODE INPUTS
# ---------------------------
diet_map = {
    "Balanced": 0.2,
    "High Sugar": 0.9,
    "High Fat": 0.8,
    "Vegetarian": 0.3,
    "High Fiber": 0.1
}

antibiotic_map = {"No": 0, "Yes": 1}

inputs = [
    age / 80,
    bmi / 40,
    sleep / 10,
    exercise / 10,
    fiber,
    sugar,
    fermented,
    diet_map[diet],
    antibiotic_map[antibiotics]
]

# ---------------------------
# PREDICTION BUTTON
# ---------------------------
if st.button("🔍 Analyze Gut Health"):

    NUM_FEATURES = model.n_features_in_

    while len(inputs) < NUM_FEATURES:
        inputs.append(0.5)

    X_input = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    # ---------------------------
    # RESULT DISPLAY
    # ---------------------------
    st.header("📊 Health Pattern Result")

    if prediction == 1:
        st.error("⚠️ Gut Pattern: High Risk")
    else:
        st.success("✅ Gut Pattern: Stable")

    st.progress(int(prob * 100))
    st.write(f"**Confidence Score:** {round(prob,2)}")

    # ---------------------------
    # VISUAL 1 – INPUT RADAR
    # ---------------------------
    st.subheader("Input Pattern Visualization")

    fig, ax = plt.subplots()
    ax.bar(
        ["Age","BMI","Sleep","Exercise","Fiber","Sugar","Fermented"],
        inputs[:7]
    )
    st.pyplot(fig)

    # ---------------------------
    # VISUAL 2 – MICROBIOME BALANCE
    # ---------------------------
    st.subheader("Microbiome Balance Simulation")

    healthy = 1 - prob
    risk = prob

    fig2, ax2 = plt.subplots()
    ax2.pie(
        [healthy, risk],
        labels=["Healthy Balance","Imbalance"],
        autopct='%1.1f%%'
    )
    st.pyplot(fig2)

    # ---------------------------
    # PERSONALIZED ADVICE
    # ---------------------------
    st.header("🧠 Personalized Recommendations")

    if prediction == 1:
        st.write("""
        - Increase fiber intake  
        - Reduce sugar  
        - Add probiotics / yogurt  
        - Maintain 7–8 hrs sleep  
        - Avoid unnecessary antibiotics  
        """)
    else:
        st.write("""
        - Maintain current diet  
        - Continue exercise  
        - Monitor sugar levels  
        - Include fermented foods weekly  
        """)

    st.info("This AI supports lifestyle guidance, not medical diagnosis.")

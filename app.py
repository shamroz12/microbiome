import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------- LOAD MODEL ----------------
model = joblib.load("microbiome_model_final.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Personalized Gut Health AI", layout="wide")

# ---------------- HEADER ----------------
st.title("AI-Driven Personalized Gut Health Simulator")
st.write("""
This interactive tool demonstrates how **lifestyle and diet patterns**
can influence gut microbiome balance using Artificial Intelligence.
It is designed for **research and educational purposes only**, not medical diagnosis.
""")

# ---------------- EXPLANATION ----------------
with st.expander("What data am I entering?"):
    st.write("""
You are entering **lifestyle and dietary indicators** that indirectly
influence gut bacteria.  

Each slider ranges from **0 (Very Low)** to **1 (Very High)**.  

The AI model converts these values into a simulated gut microbial balance
and provides preventive health suggestions.
""")

# ---------------- USER INPUTS ----------------
st.sidebar.header("Lifestyle & Gut Health Indicators")

features = [
    ("Daily Fiber Intake", "How much fruits, vegetables, whole grains you consume"),
    ("Fermented Food Consumption", "Yogurt, kefir, kimchi, etc."),
    ("Recent Antibiotic Usage", "Higher value means more recent/frequent use"),
    ("Sugar Intake Level", "High sugar negatively impacts gut bacteria"),
    ("Probiotic Consumption", "Supplements or probiotic foods"),
    ("Stress Level", "Higher stress can disturb gut balance"),
    ("Sleep Quality", "Better sleep improves microbiome health"),
    ("Physical Activity", "Exercise supports microbial diversity"),
    ("Processed Food Intake", "High processed food harms gut bacteria"),
    ("Digestive Symptom Frequency", "Bloating, discomfort, irregularity")
]

inputs = []

for name, description in features:
    val = st.sidebar.slider(name, 0.0, 1.0, 0.5)
    st.sidebar.caption(description)
    inputs.append(val)

# ---------------- PREDICTION ----------------
NUM_FEATURES = model.n_features_in_

# Pad remaining features with 0.5
while len(inputs) < NUM_FEATURES:
    inputs.append(0.5)

X_input = np.array(inputs).reshape(1, -1)
X_scaled = scaler.transform(X_input)

    # ---------------- PERSONALIZED SUGGESTIONS ----------------
    st.subheader("Personalized Preventive Suggestions")

    if prob < 0.4:
        st.write("""
- Increase **fiber intake** (fruits, vegetables, legumes)  
- Add **probiotic foods** like yogurt or kefir  
- Reduce **processed food and sugar**  
- Improve **sleep routine**  
""")
    elif prob < 0.7:
        st.write("""
- Maintain balanced diet  
- Increase **physical activity**  
- Monitor stress levels  
- Continue probiotic foods occasionally  
""")
    else:
        st.write("""
- Your lifestyle currently supports **good gut stability**  
- Continue balanced diet and exercise  
- Maintain healthy sleep and stress habits  
""")

    # ---------------- EXPLAINABLE AI ----------------
    if hasattr(model, "feature_importances_"):
        st.subheader("Key Factors Influencing AI Decision")

        importances = model.feature_importances_
        feature_labels = [f[0] for f in features]

        feat_df = pd.DataFrame({
            "Factor": feature_labels,
            "Importance": importances[:len(feature_labels)]
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(feat_df.set_index("Factor"))

# ---------------- DISCLAIMER ----------------
st.info("""
This platform is a **computational simulation tool**.
It does not replace clinical consultation.
Its purpose is to demonstrate how AI can support **preventive personalized healthcare engineering**.
""")

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------- LOAD MODEL ----------------
model = joblib.load("microbiome_model_final.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="AI Microbiome Health Tool", layout="wide")

# ---------------- BANNER ----------------
st.image("https://images.unsplash.com/photo-1581091870627-3c4e3f4a1a2b", use_column_width=True)

st.title("AI-Driven Gut Microbiome Health Assessment")
st.write("This interactive tool simulates gut microbial balance and predicts microbiome stability patterns using Artificial Intelligence.")

# ---------------- USER EXPLANATION ----------------
with st.expander("What are these inputs?"):
    st.write("""
    These sliders represent **relative microbial indicators** inside the human gut.
    Since real microbiome sequencing data is complex, this demo uses normalized
    values between 0 and 1.

    **Example:**
    - 0.2 → Low presence
    - 0.5 → Moderate presence
    - 0.8 → High presence

    This tool is for **educational and research demonstration only**, not medical diagnosis.
    """)

# ---------------- FEATURE NAMES ----------------
feature_names = [
    "Firmicutes Ratio",
    "Bacteroidetes Ratio",
    "Proteobacteria Presence",
    "Actinobacteria Level",
    "Microbial Diversity Score",
    "Inflammation Marker Proxy",
    "Short Chain Fatty Acid Level",
    "Pathogen Load Indicator",
    "Beneficial Bacteria Index",
    "Gut Stability Score"
]

st.sidebar.header("Microbiome Indicators (User Inputs)")

NUM_FEATURES = model.n_features_in_
inputs = []

for i in range(NUM_FEATURES):
    name = feature_names[i] if i < len(feature_names) else f"Microbial Factor {i+1}"
    val = st.sidebar.slider(name, 0.0, 1.0, 0.5)
    inputs.append(val)

# ---------------- EXAMPLE SCENARIO ----------------
st.sidebar.markdown("### Example Scenario")
st.sidebar.write("""
Healthy Adult Example:
- Firmicutes Ratio → 0.6
- Diversity Score → 0.8
- Pathogen Load → 0.2
""")

# ---------------- PREDICTION ----------------
if st.button("Analyze Microbiome Pattern"):
    X_input = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    st.subheader("Health Pattern Result")

    if prob > 0.75:
        st.success("Microbiome Pattern: Stable")
    elif prob > 0.5:
        st.warning("Microbiome Pattern: Moderate Risk")
    else:
        st.error("Microbiome Pattern: High Risk")

    st.progress(int(prob * 100))
    st.write(f"Confidence Score: {prob:.2f}")

    # ---------------- EXPLAINABLE AI ----------------
    if hasattr(model, "feature_importances_"):
        st.subheader("Top Influencing Microbial Factors")

        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Factor": [f"Factor {i+1}" for i in range(len(importances))],
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(5)

        st.bar_chart(feat_df.set_index("Factor"))

# ---------------- DISCLAIMER ----------------
st.info("Note: This is a computational simulation tool for research demonstration purposes only.")

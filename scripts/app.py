import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Sequence Disruption Risk Dashboard", layout="centered")
st.title("Sequence Disruption Risk Dashboard")

st.write("Current working directory:", os.getcwd())
st.write("Files in this folder:", os.listdir())

try:
    model = joblib.load("logistic_spoilage_model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    st.success("Model files loaded successfully")
    st.write("Feature columns:", feature_cols)
    st.write("Model classes:", getattr(model, "classes_", "No classes_ found"))
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.markdown(
    """
This dashboard predicts whether a sequence is **low risk** or **elevated risk**
for operational disruption based on a few sequence-level features.
"""
)

st.header("Enter Sequence Information")

total_blocked_hrs = st.number_input("Total Blocked Hours", min_value=0.0, max_value=100.0, value=11.5, step=0.1)
seq_ttl_legs = st.number_input("Total Legs in Sequence", min_value=0, max_value=20, value=3, step=1)
layover = st.number_input("Layover", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
seq_start_hrs = st.number_input("Sequence Start Hour", min_value=0.0, max_value=24.0, value=11.0, step=0.1)
assignment_count = st.number_input("Assignment Count", min_value=0, max_value=500, value=10, step=1)

input_df = pd.DataFrame([{
    "TOTAL_BLOCKED_HRS": total_blocked_hrs,
    "SEQ_TTL_LEGS": seq_ttl_legs,
    "LAYOVER": layover,
    "SEQ_START_HRS": seq_start_hrs,
    "ASSIGNMENT_COUNT": assignment_count
}])

try:
    input_df = input_df[feature_cols]
    st.write("Input dataframe:", input_df)
except Exception as e:
    st.error(f"Feature mismatch error: {e}")
    st.write("Input columns:", input_df.columns.tolist())
    st.write("Expected columns:", feature_cols)
    st.stop()

if st.button("Predict Risk"):
    try:
        prob_risk = model.predict_proba(input_df)[0, 1]
        pred_class = int(prob_risk >= 0.5)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    if prob_risk < 0.35:
        risk_level = "Low Risk"
        recommendation = "No immediate operational action needed."
        color = "green"
    elif prob_risk < 0.60:
        risk_level = "Medium Risk"
        recommendation = "Monitor sequence and review downstream connections."
        color = "orange"
    else:
        risk_level = "High Risk"
        recommendation = "Flag for proactive operational review."
        color = "red"

    st.header("Prediction Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Class", "Elevated Risk" if pred_class == 1 else "Low Risk")
    with col2:
        st.metric("Risk Probability", f"{prob_risk:.2%}")

    st.markdown(f"### Risk Level: :{color}[{risk_level}]")
    st.info(recommendation)
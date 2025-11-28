import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------ SETUP ------------------
PIPE = Path("models/model.pkl")

st.set_page_config(page_title="Event Attendance Predictor", layout="wide")
st.title("📊 FASA Event Attendance Predictor")

if show_shap:
    st.subheader("📈 SHAP: Why This Prediction?")

    # SHAP calculation
    transformed = pipeline["pre"].transform(input_df)
    shap_values = explainer(transformed)

    # Bar plot of SHAP importance
    fig, ax = plt.subplots(figsize=(8,5))
    shap.summary_plot(shap_values.values, input_df, plot_type="bar", show=False)
    st.pyplot(fig)



# ------------------ MODEL LOADING ------------------
if not PIPE.exists():
    st.error("❌ Model not found.\nRun this first in your terminal:\n\n`python3 src/train_model.py`")
    st.stop()

pipeline = joblib.load(PIPE)

# Extract model + preprocessor safely
model = pipeline["model"]
preprocessor = pipeline["pre"]

try:
    explainer = shap.TreeExplainer(model)
except Exception:
    explainer = None
    st.warning("⚠️ SHAP explainer not supported for this model. Explanation disabled.")


# ------------------ USER INPUT ------------------
st.header("🎯 Predict Attendance")

col1, col2 = st.columns(2)

with col1:
    event_type = st.selectbox("Event Type", ["GBM", "Fundraiser", "Collab", "Practice", "Special"])
    date_input = st.date_input("Event Date")
    likes = st.number_input("Instagram Likes", value=50, min_value=0)
    comments = st.number_input("Instagram Comments", value=5, min_value=0)

with col2:
    has_food = st.checkbox("Food Served?")
    is_collab = st.checkbox("Collaboration?")
    show_shap = st.checkbox("Show SHAP Explainability")

day_of_week = pd.to_datetime(date_input).day_name()
days_into_semester = (pd.to_datetime(date_input) - pd.to_datetime("2025-08-28")).days

input_df = pd.DataFrame([{
    "event_type": event_type,
    "likes": likes,
    "comments": comments,
    "day_of_week": day_of_week,
    "days_into_semester": days_into_semester,
    "has_food": int(has_food),
    "is_collab": int(is_collab),
}])


# ------------------ PREDICTION BUTTON ------------------
if st.button("Predict"):
    pred = pipeline.predict(input_df)[0]
    st.success(f"Estimated Attendance: **{int(pred)} people**")

    # ---------------- SHAP PLOT ----------------
    if show_shap and explainer is not None:
        st.subheader("📈 SHAP: Why This Prediction?")

        X_pre = preprocessor.transform(input_df)
        shap_values = explainer.shap_values(X_pre)

        fig, ax = plt.subplots(figsize=(8,5))
        shap.summary_plot(
            shap_values,
            X_pre,
            feature_names=input_df.columns,
            plot_type="bar",
            show=False
        )
        st.pyplot(fig)

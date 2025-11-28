import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------- SETUP ----------------
PIPE = Path("models/model.pkl")

st.set_page_config(page_title="Event Attendance Predictor", layout="wide")
st.title("📊 FASA Event Attendance Predictor")

# ---------------- MODEL LOADING ----------------
if not PIPE.exists():
    st.error("❌ Model not found.\nRun this first:\n\n`python3 src/train_model.py`")
    st.stop()

pipeline = joblib.load(PIPE)
model = pipeline["model"]
preprocessor = pipeline["pre"]

# ---------------- USER INPUT ----------------
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
    show_importance = st.checkbox("Show Feature Importance Explanation")

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

# ---------------- PREDICTION ----------------
if st.button("Predict"):
    pred = pipeline.predict(input_df)[0]
    st.success(f"Estimated Attendance: **{int(pred)} people**")

    # ---------------- FEATURE IMPORTANCE (SHAP ALTERNATIVE) ----------------
    if show_importance:
        st.subheader("📈 Feature Importance (What The Model Thinks Matters Most)")

        # Get transformed column names
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = list(input_df.columns)

        # Get model importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            st.warning("This model does not support feature importance.")
            st.stop()

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # Plot
        fig, ax = plt.subplots(figsize=(8,5))
        ax.barh(importance_df["Feature"], importance_df["Importance"])
        ax.invert_yaxis()
        ax.set_title("Feature Importance Ranking")
        st.pyplot(fig)

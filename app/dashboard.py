import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="FASA Event Attendance Predictor", layout="wide")

MODEL_PATH = Path("models/model.pkl")
CLEANED_DATA = Path("data/events_cleaned.csv")

st.title("📊 FASA Event Attendance Dashboard")

# ------------------ LOAD MODEL ------------------
if not MODEL_PATH.exists():
    st.error("❌ Model missing. Run: python3 src/train_model.py")
    st.stop()

pipeline = joblib.load(MODEL_PATH)

model = pipeline["model"]
pre = pipeline["pre"]

# ------------------ LOAD DATA ------------------
if CLEANED_DATA.exists():
    df = pd.read_csv(CLEANED_DATA)
    df["date"] = pd.to_datetime(df["date"])
else:
    df = pd.DataFrame()

# ====================== TABS ==========================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Attendance",
    "📈 Trends & Analytics",
    "📂 Data Explorer",
    "🧠 Model Insights"
])

# -------------------------------------------------------
# TAB 1 — PREDICTOR
# -------------------------------------------------------
with tab1:
    st.header("🔮 Event Attendance Predictor")

    col1, col2 = st.columns(2)

    with col1:
        event_type = st.selectbox("Event Type", ["GBM", "Fundraiser", "Collab", "Practice", "Special"])
        date_input = st.date_input("Event Date")
        likes = st.number_input("Instagram Likes", min_value=0, value=50)
        comments = st.number_input("Instagram Comments", min_value=0, value=5)

    with col2:
        has_food = st.checkbox("Food Served?")
        is_collab = st.checkbox("Collaboration Event?")
    
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

    if st.button("Predict Attendance"):
        pred = pipeline.predict(input_df)[0]

        st.success(f"🎉 Expected Attendance: **{int(pred)} people**")

        # Suggestion Engine
        suggestions = []
        if likes < 50:
            suggestions.append("Increase marketing / reel posts to raise likes 📢")
        if not has_food:
            suggestions.append("Add food to increase turnout 🍜")
        if not is_collab:
            suggestions.append("Collaborate with another org 🤝")

        st.subheader("💡 Suggestions to Improve Attendance")
        if suggestions:
            for s in suggestions:
                st.write(f"• {s}")
        else:
            st.write("Your event looks strong! 👍")


# -------------------------------------------------------
# TAB 2 — TRENDS
# -------------------------------------------------------
with tab2:
    st.header("📈 Event Trends & Analytics")

    if df.empty:
        st.warning("No event history loaded.")
    else:
        colA, colB = st.columns(2)

        # Attendance over time
        with colA:
            st.subheader("📅 Attendance Over Time")
            fig, ax = plt.subplots()
            ax.plot(df["date"], df["attendance"], linewidth=2)
            ax.set_xlabel("Date")
            ax.set_ylabel("Attendance")
            st.pyplot(fig)

        # Attendance by event type
        with colB:
            st.subheader("🏷️ Attendance by Event Type")
            type_avg = df.groupby("event_type")["attendance"].mean()
            fig, ax = plt.subplots()
            type_avg.plot(kind="bar", ax=ax)
            ax.set_ylabel("Avg Attendance")
            st.pyplot(fig)


# -------------------------------------------------------
# TAB 3 — DATA EXPLORER
# -------------------------------------------------------
with tab3:
    st.header("📂 Explore Events Data")
    if df.empty:
        st.info("No event data available.")
    else:
        st.dataframe(df)


# -------------------------------------------------------
# TAB 4 — MODEL INSIGHTS
# -------------------------------------------------------
with tab4:
    st.header("🧠 Model Feature Importance")

    # Pull feature importance
    importances = model.feature_importances_
    feature_names = pre.get_feature_names_out()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    st.dataframe(importance_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.invert_yaxis()
    st.pyplot(fig)

![Event Engagement Predictor Banner](https://raw.githubusercontent.com/dominictrojillo/event-engagement-predictor/main/assets/banner.png)
# FASA Event Engagement Predictor  
**Author:** Dominic Trojillo  
**Repository:** https://github.com/dominictrojillo/event-engagement-predictor  
**Live Application:**
https://event-engagement-predictor-qgsq8isudwn2o4ynxcwr3d.streamlit.app

A machine learning application designed to forecast attendance for student organization events using historical event logs, social media engagement, and event metadata.
Built for the Filipino-American Student Association (FASA) at UMBC and deployed as an interactive Streamlit dashboard.

---

## 📌 Overview  
This project provides a data-driven approach to estimating event turnout by analyzing factors such as event type, marketing engagement, weekday trends, and timeline within the semester.
It includes a machine learning model trained on a mix of historical and synthetic data, as well as a user-friendly interface for making predictions and reviewing event insights.

This tool provides:  
✔ An ML-powered attendance prediction  
✔ Feature importance visualization (no SHAP required)
✔ A polished Streamlit dashboard  
✔ Automatically cleaned + augmented synthetic data to improve stability  
✔ A realistic dataset built from UMBC student org behavior  

This project is designed to be a **real, production-ready, portfolio-grade ML application**.

---

## 🖥 Features  
### Prediction Engine
Estimates attendance based on:
  - Event type
  - Instagram likes and comments
  - Whether the event includes food
  - Whether the event is a collaboration
  - Day of week and days into the semester

### Dataset Construction
  - Combines real FASA historical data with more than 200 synthetic events
  - Synthetic events follow realistic distributions based on UMBC organization behavior
  - Includes preprocessing pipeline and data cleaning scripts

### Machine Learning Model
  - Random Forest Regressor
  - One-hot encoding of categorical features
  - Train/test split with cross-validation for stability
  - Feature importance analysis included in dashboard

### Streamlit Dashboard
  - Real-time attendance prediction
  - Event analytics and visualizations
  - Clean, minimal, responsive interface
  - Fully deployable on Streamlit Cloud
---

## 📁 Project Structure
  event-engagement-predictor/ <br/>
  │ <br/>
  ├── app/ <br/>
  │ └── dashboard.py <br/>
  │ <br/>
  ├── src/ <br/>
  │ └── train_model.py <br/>
  │ <br/>
  ├── data/ <br/>
  │ ├── events.csv <br/>
  │ ├── events_cleaned.csv <br/>
  │ ├── events_source_merged.csv <br/>
  │ └── synthetic_events.csv <br/>
  │ <br/>
  ├── models/ <br/>
  │ └── model.pkl <br/>
  │ <br/>
  ├── scripts/ <br/>
  │ └── generate_synthetic.py <br/>
  │ <br/>
  ├── README.md <br/>
  ├── requirements.txt <br/>
  └── .gitignore <br/>

---

## 🚀 Getting Started

### 1 Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

### 2 Install dependencies
pip install -r requirements.txt

### 3 Train the model
python3 src/train_model.py

### 4 Run Streamlit dashboard
streamlit run app/dashboard.py


## 📸 Screenshot Gallery

Replace each placeholder with a screenshot later

🎛 Main Prediction Interface

Place Screenshot Here (placeholder)

📈 SHAP Explainability

Place Screenshot Here (placeholder)

📊 Training Data Explorer

Place Screenshot Here (placeholder)

---

### Deployment Notes
Streamlit Cloud runs in a clean environment and does not automatically include local files.
For the dashboard to function on Streamlit Cloud, ensure:
models/model.pkl is committed to the repository, or
The model is retrained automatically if missing.

### Future Improvements
- Expand dataset with additional academic-year trends
- Integrate Instagram API for automated data collection
- Add temporal forecasting for multi-event planning
- Support exportable event planning reports

# 🎓 FASA Event Engagement Predictor  
### Predict Attendance for Student Organization Events Using Machine Learning  
**Author:** Dominic Trujillo  
**Repo:** https://github.com/dominictrojillo/event-engagement-predictor  

---

## 📌 Overview  
The **Event Engagement Predictor** is a machine learning project designed to help UMBC’s Filipino-American Student Association (FASA) estimate event attendance using historical data, social media performance, and event characteristics.

This tool provides:  
✔ An ML-powered attendance prediction  
✔ Event insights & SHAP explainability  
✔ A polished Streamlit dashboard  
✔ Automatically cleaned + augmented synthetic data to improve stability  
✔ A realistic dataset built from UMBC student org behavior  

This project is designed to be a **real, production-ready, portfolio-grade ML application**.

---

## 🖥 Features  
### 🎯 **Core Features**
- Predict expected event attendance
- Input:
  - Event Type (GBM, Fundraiser, Practice, Special, Collab)
  - Instagram Likes + Comments  
  - Collaboration indicator  
  - Food indicator  
  - Date → auto-generates weekday + semester progression
- Trained on a mix of:
  - Real FASA data  
  - Synthetic, noise-controlled event logs (200+ events)

### 🔍 **Explainability (SHAP)**
- Shows which factors boosted or reduced attendance  
- Helps boards plan better events with data-driven decisions

### 📊 **Clean UI**
- Three dashboard sections:
  1. **Prediction Panel**
  2. **Event Insights & SHAP Visualization**
  3. **Historical Trends & Training Data Explorer**

---

## 📁 Project Structure
event-engagement-predictor/
│
├── app/
│ └── dashboard.py
│
├── src/
│ └── train_model.py
│
├── data/
│ ├── events.csv
│ ├── events_cleaned.csv
│ ├── events_source_merged.csv
│ └── synthetic_events.csv
│
├── models/
│ └── model.pkl
│
├── scripts/
│ └── generate_synthetic.py
│
├── README.md
├── requirements.txt
└── .gitignore

---

## 🚀 Getting Started

### 1 Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate

### 2 Install dependencies
pip install -r requirements.txt

### 3 Train the model
python3 src/train_model.py

###4 Run Streamlit dashboard
streamlit run app/dashboard.py


📸 Screenshot Gallery

Replace each placeholder with a screenshot later

🎛 Main Prediction Interface

Place Screenshot Here (placeholder)

📈 SHAP Explainability

Place Screenshot Here (placeholder)

📊 Training Data Explorer

Place Screenshot Here (placeholder)



🤖 Machine Learning Details
Model: Random Forest Regressor
Encoding: OneHotEncoder for categorical features
Synthetic data generation with realistic FASA-style distributions:
Attendance curves
Engagement scaling
Event-type multipliers
Weekday differences
Collab multipliers
Food multipliers

Target variable:
attendance

Feature set:
event_type
likes
comments
day_of_week
days_into_semester
has_food
is_collab

---

# 📦 **2. requirements.txt**
Create this file in your project root:

```txt
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
streamlit==1.32.0
joblib==1.3.2
python-dateutil==2.9.0.post0
shap==0.45.0
matplotlib==3.8.3

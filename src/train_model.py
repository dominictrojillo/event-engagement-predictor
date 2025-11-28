# src/train_model.py
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from dateutil import parser as dtparser

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Try XGBoost; fallback to RandomForest
try:
    from xgboost import XGBRegressor
    MODEL_CLASS = "xgb"
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    MODEL_CLASS = "rf"

ROOT = Path.cwd()
DATA_IN = ROOT / "data" / "events.csv"
DATA_CLEANED = ROOT / "data" / "events_cleaned.csv"
MODELS_DIR = ROOT / "models"
PIPE_PATH = MODELS_DIR / "pipeline.pkl"
FI_CSV = MODELS_DIR / "feature_importances.csv"
os.makedirs(MODELS_DIR, exist_ok=True)

if not DATA_IN.exists():
    raise FileNotFoundError(f"Missing input CSV: {DATA_IN}. Run the synthetic generator script first if you haven't.")

# Load data
raw = pd.read_csv(DATA_IN, keep_default_na=False)

# Normalize column names
col_map = {}
for c in raw.columns:
    lc = c.strip().lower()
    if "date" in lc:
        col_map[c] = "date_raw"
    elif "event" in lc and "name" in lc:
        col_map[c] = "event_name"
    elif "attendance" in lc:
        col_map[c] = "attendance"
    elif "like" in lc:
        col_map[c] = "likes"
    elif "comment" in lc:
        col_map[c] = "comments"
    elif "location" in lc:
        col_map[c] = "location"
    elif "note" in lc or "logistic" in lc or "details" in lc:
        col_map[c] = "notes"
    else:
        col_map[c] = c.strip()

raw = raw.rename(columns=col_map)

# parse dates
def tryparse(s):
    try:
        if pd.isna(s) or str(s).strip() == "":
            return pd.NaT
        return dtparser.parse(str(s), fuzzy=True)
    except Exception:
        return pd.NaT

raw["date"] = raw.get("date_raw","").apply(tryparse)
raw = raw[~raw["date"].isna()].copy()
if raw.empty:
    raise SystemExit("[ERROR] no parseable dates in input csv.")

# normalize numeric columns
for col in ["attendance","likes","comments"]:
    if col not in raw.columns:
        raw[col] = np.nan
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

# Feature engineering
df = raw.copy()
df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.day_name()
df["month"] = df["date"].dt.month
df["week"] = df["date"].dt.isocalendar().week.astype(int)
df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
df["days_into_semester"] = (df["date"] - df["date"].min()).dt.days.astype(int)

# heuristics
df["is_major"] = (
    df["event_name"].fillna("").str.contains("pistahan|pistahan|showcase|mega|festival|i-fest", case=False, na=False)
    | df.get("notes","").fillna("").str.contains("pistahan|showcase|major|festival|alumni", case=False, na=False)
).astype(int)
df["has_food"] = df.get("notes","").fillna("").str.contains("food|halo|taho|ice cream|snack|lunch|dinner", case=False, na=False).astype(int)
df["is_collab"] = (
    df.get("event_type","").fillna("").str.contains("collab|collaboration|x ", case=False, na=False)
    | df.get("notes","").fillna("").str.contains("hosted with|collab|collaboration", case=False, na=False)
).astype(int)

# fill missing likes/comments with small values to avoid NAs
df["likes"] = df["likes"].fillna(df["likes"].median() if not df["likes"].isna().all() else 30)
df["comments"] = df["comments"].fillna(df["comments"].median() if not df["comments"].isna().all() else 3)

# target
df["attendance"] = df["attendance"].fillna(df["attendance"].median() if not df["attendance"].isna().all() else 50).astype(int)

# features
features = [
    "event_type","location","likes","comments","day_of_week",
    "days_into_semester","month","is_weekend","is_major","has_food","is_collab"
]

X = df[features]
y = df["attendance"]

# Preprocessor
categorical = ["event_type","location","day_of_week"]
numeric = ["likes","comments","days_into_semester","month","is_weekend","is_major","has_food","is_collab"]

ohe_kwargs = {"handle_unknown":"ignore"}
try:
    from sklearn.preprocessing import OneHotEncoder as _OHE
    ohe = _OHE(**ohe_kwargs, sparse_output=False)
except Exception:
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(**ohe_kwargs, sparse=False)

from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer([
    ("cat", ohe, categorical),
    ("num", "passthrough", numeric)
], remainder="drop")

# Choose model
if MODEL_CLASS == "xgb":
    model = XGBRegressor(n_estimators=300, random_state=42, tree_method="hist", verbosity=0)
else:
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=400, random_state=42)

pipeline = Pipeline([
    ("pre", preprocessor),
    ("model", model)
])

# Train/test split
test_size = 0.2 if len(df) >= 50 else 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"[RESULT] Test MAE: {mae:.2f} (train={len(X_train)}, test={len(X_test)})")

# Cross-val if enough data
if len(X_train) >= 20:
    cv = min(5, len(X_train))
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
    print(f"[RESULT] CV MAE: {(-cv_scores.mean()):.2f}")

# Save pipeline and cleaned data
joblib.dump(pipeline, PIPE_PATH)
df.to_csv(DATA_CLEANED, index=False)
print(f"[SAVED] Pipeline: {PIPE_PATH} | Cleaned data: {DATA_CLEANED}")

# feature importances (attempt)
try:
    model_inst = pipeline.named_steps["model"]
    # build feature names
    cat_names = []
    try:
        cat_names = list(pipeline.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(categorical))
    except Exception:
        # fallback
        for c in categorical:
            vals = df[c].fillna("___NA___").unique().tolist()
            cat_names += [f"{c}_{v}" for v in vals]
    feat_names = cat_names + numeric
    importances = model_inst.feature_importances_
    fi = pd.DataFrame({"feature": feat_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False)
    fi.to_csv(FI_CSV, index=False)
    print(f"[SAVED] Feature importances to {FI_CSV}")
except Exception as e:
    print(f"[WARN] could not compute feature importances: {e}")

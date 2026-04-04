from fastapi import FastAPI
import pandas as pd
import joblib


# Initialize app
app = FastAPI()

# Load model, scaler, features
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "models", "features.pkl"))

@app.post("/predict")
def predict(data: dict):

    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # Apply encoding
    df = pd.get_dummies(df)

    # Align columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scaling
    df = scaler.transform(df)

    # Prediction
    proba = model.predict_proba(df)[0][1]

    # Risk logic
    if proba < 0.3:
        risk = "Low"
    elif proba < 0.7:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "churn_probability": float(proba),
        "risk_level": risk
    }
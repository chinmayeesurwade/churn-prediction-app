import pandas as pd
import joblib

# ==============================
# Load model, scaler, features
# ==============================
model = joblib.load("../models/model.pkl")
scaler = joblib.load("../models/scaler.pkl")
feature_columns = joblib.load("../models/features.pkl")

# ==============================
# Prediction Function
# ==============================
def predict(input_data):

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # Apply same encoding
    df = pd.get_dummies(df)

    # Align columns with training data
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Apply scaling
    df = scaler.transform(df)

    # Predict probability
    proba = model.predict_proba(df)[0][1]

    # Convert to risk level
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


# ==============================
# Test Prediction
# ==============================
if __name__ == "__main__":

    sample_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70,
        "TotalCharges": 350
    }

    result = predict(sample_customer)

    print(result)
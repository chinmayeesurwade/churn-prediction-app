import streamlit as st
import pandas as pd
import joblib
import os

# ==============================
# LOAD MODEL + FILES
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "models", "features.pkl"))

# ==============================
# UI TITLE
# ==============================

st.title("Customer Churn Prediction System")
st.caption("End-to-End ML Project | Built with Python, Scikit-learn, FastAPI, Streamlit")

st.write("### Enter Customer Details:")

# ==============================
# INPUT FIELDS
# ==============================

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No"])

StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

MonthlyCharges = st.number_input("Monthly Charges", 0.0)
TotalCharges = st.number_input("Total Charges", 0.0)

# ==============================
# PREDICTION BUTTON
# ==============================

if st.button("Predict"):

    # Create input dictionary
    input_data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Encoding
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

    # ==============================
    # OUTPUT (COLORED)
    # ==============================

    st.write("## Result")

    if risk == "High":
        st.error(f" High Risk Customer (Churn Probability: {proba:.2f})")
    elif risk == "Medium":
        st.warning(f" Medium Risk Customer (Churn Probability: {proba:.2f})")
    else:
        st.success(f" Low Risk Customer (Churn Probability: {proba:.2f})")

    # ==============================
    # BUSINESS INSIGHTS
    # ==============================

    st.subheader(" Recommendation")

    if risk == "High":
        st.write(" Immediate action required: Offer discounts, retention calls, or special plans.")
    elif risk == "Medium":
        st.write(" Engage customer with personalized offers and communication.")
    else:
        st.write("Maintain satisfaction with loyalty programs and good service.")

    # ==============================
    # EXTRA: SHOW PROBABILITY BAR
    # ==============================

    st.progress(float(proba))

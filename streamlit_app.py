import streamlit as st
import pandas as pd
import joblib
import os

# ==============================
# BASE PATH
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================
# LOAD MODELS SAFELY
# ==============================
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
features_path = os.path.join(BASE_DIR, "models", "features.pkl")

if not os.path.exists(model_path):
    st.error("model.pkl not found in models folder")
    st.stop()

if not os.path.exists(scaler_path):
    st.error("scaler.pkl not found in models folder")
    st.stop()

if not os.path.exists(features_path):
    st.error(" features.pkl not found in models folder")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_columns = joblib.load(features_path)

# ==============================
# UI TITLE
# ==============================
st.title("Customer Churn Prediction System")
st.caption("End-to-End ML Project | Built with Python & Streamlit")

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

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)

# ==============================
# PREDICTION
# ==============================
if st.button("Predict"):

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

    df = pd.DataFrame([input_data])

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align columns with training data
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scaling
    df = scaler.transform(df)

    # Prediction
    proba = model.predict_proba(df)[0][1]

    # Risk classification
    if proba < 0.3:
        risk = "Low"
    elif proba < 0.7:
        risk = "Medium"
    else:
        risk = "High"

    # ==============================
    # OUTPUT
    # ==============================
    st.write("## Result")

    if risk == "High":
        st.error(f"High Risk Customer (Churn Probability: {proba:.2f})")
    elif risk == "Medium":
        st.warning(f"Medium Risk Customer (Churn Probability: {proba:.2f})")
    else:
        st.success(f"Low Risk Customer (Churn Probability: {proba:.2f})")

    # Progress bar
    st.progress(float(proba))

    # Recommendation
    st.subheader("Recommendation")

    if risk == "High":
        st.write("Offer discounts, retention calls, or special plans.")
    elif risk == "Medium":
        st.write("Engage with personalized offers.")
    else:
        st.write("Maintain customer satisfaction.")

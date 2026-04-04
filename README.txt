#  Customer Churn Prediction System

##  Executive Summary

Customer churn directly impacts revenue and customer lifetime value.
This project builds a production-ready machine learning system to **identify high-risk customers early and enable targeted retention strategies**.

Instead of just predicting churn, the system translates model outputs into **actionable business decisions**.

---

##  Business Problem

Telecom companies lose significant revenue due to customer churn.
The key challenge is not just identifying churners, but **detecting them early enough to intervene effectively**.

---

##  Solution Approach

This system:

* Predicts churn probability using customer behavioral and billing data
* Segments customers into **Low, Medium, High risk categories**
* Provides **business-driven recommendations** for each segment

 Focus is on **decision-making, not just prediction**

---

##  Key Insights from Data

### 1. Contract Type is the strongest churn driver

* Month-to-month users show significantly higher churn
* Long-term contracts drastically reduce churn

 Insight: **Lock-in strategies improve retention**

---

### 2. High Monthly Charges → Higher churn risk

* Customers paying more are more likely to leave

 Insight: **Perceived value mismatch → pricing sensitivity**

---

### 3. Fiber Optic users churn more

* Despite better service, churn is higher

 Insight: **Service quality alone ≠ retention**
(Customer expectations are higher)

---

### 4. Short tenure customers are high-risk

* New customers churn quickly

 Insight: **Onboarding experience is critical**

---

## ⚙️ System Architecture

```id="arch123"
User Input → Streamlit UI → Preprocessing → ML Model → Prediction → Risk Segmentation → Business Action
                                ↓
                           Saved Artifacts
                     (Model, Scaler, Features)
```

---

##  Project Structure

```id="struct456"
Churn Prediction/
│
├── app/
│   ├── streamlit_app.py   # Interactive Dashboard
│   ├── api.py             # FastAPI backend
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│
├── src/                   # Training pipeline
├── data/
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn (Logistic Regression)
* FastAPI (API layer)
* Streamlit (UI dashboard)

---

## 📊 Model Performance

| Metric            | Value                           |
| ----------------- | ------------------------------- |
| Accuracy          | ~80%                            |
| ROC-AUC           | ~0.83                           |
| Recall (Churners) | Improved using threshold tuning |

 Business trade-off:

* Higher recall for churners → better retention targeting
* Slight drop in precision → acceptable in business context

---

##  Risk Segmentation Strategy

| Risk Level | Probability Range | Action                           |
| ---------- | ----------------- | -------------------------------- |
| Low        | < 0.2             | Maintain engagement              |
| Medium     | 0.2 – 0.6         | Targeted offers                  |
| High       | > 0.6             | Immediate retention intervention |

---

##  Key Engineering Decisions

### ✔ Feature Engineering

* One-hot encoding for categorical variables
* Removed customerID (non-informative feature)

### ✔ Scaling

* StandardScaler used to stabilize model convergence

### ✔ Model Choice

* Logistic Regression selected for:

  * Interpretability
  * Fast inference
  * Baseline benchmarking

---

##  Features

* Real-time churn prediction
* Interactive UI for business users
* Risk categorization
* Actionable recommendations
* API-ready backend

---

##  Run Locally

### Install dependencies

```id="run1"
pip install -r requirements.txt
```

### Run dashboard

```id="run2"
python -m streamlit run app/streamlit_app.py
```

### Run API

```id="run3"
python -m uvicorn app.api:app --reload
```

---

##  Live Application

 (Add your Streamlit deployment link here)

---

##  Future Improvements

* Replace Logistic Regression with ensemble models (XGBoost, Random Forest)
* Add SHAP for model explainability
* Integrate real-time data pipelines
* Deploy using Docker + cloud (AWS/GCP)

---

##  Author

Chinmayee Surwade
chinmayee.surwade@gmail.com

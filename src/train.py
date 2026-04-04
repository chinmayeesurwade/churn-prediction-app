import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ==============================
# STEP 1: Load Data
# ==============================
df = pd.read_csv("../data/churn.csv")

# ==============================
# STEP 2: Data Cleaning
# ==============================

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df = df.drop(['customerID'], axis=1)

# ==============================
# STEP 3: Encoding
# ==============================

df_encoded = pd.get_dummies(df, drop_first=True)

# ==============================
# STEP 4: Features & Target
# ==============================

X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

feature_columns = X.columns

# ==============================
# STEP 5: Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# STEP 6: Train Model
# ==============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
# ==============================
# STEP 7: Evaluate
# ==============================

proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, proba)

print("ROC-AUC Score:", roc_auc)

# ==============================
# STEP 8: Save Model
# ==============================

joblib.dump(model, "../models/model.pkl")
joblib.dump(feature_columns, "../models/features.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("Model, features and scaler saved successfully!")
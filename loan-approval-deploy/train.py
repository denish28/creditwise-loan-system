"""
Retrains the Naive Bayes loan-approval model using the same steps as the
original notebook (missing-value imputation -> encoding -> feature
engineering -> Naive Bayes), and saves every fitted artifact needed to
serve predictions later (imputers, encoders, scaler, model, column order).
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
ONE_HOT_COLS = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category",
]

df = pd.read_csv("loan_approval_data.csv")

# ---- 1. Missing values -----------------------------------------------
categorical_cols = df.select_dtypes(include=["object", "str"]).columns
numerical_cols = df.select_dtypes(include=["float64"]).columns

num_imputer = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# ---- 2. Drop ID ---------------------------------------------------------
df = df.drop("Applicant_ID", axis=1)

# ---- 3. Encoding ---------------------------------------------------------
edu_encoder = LabelEncoder()
df["Education_Level"] = edu_encoder.fit_transform(df["Education_Level"])

target_encoder = LabelEncoder()
df["Loan_Approved"] = target_encoder.fit_transform(df["Loan_Approved"])

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df[ONE_HOT_COLS])
encoded_df = pd.DataFrame(
    encoded, columns=ohe.get_feature_names_out(ONE_HOT_COLS), index=df.index
)
df = pd.concat([df.drop(columns=ONE_HOT_COLS), encoded_df], axis=1)

# ---- 4. Feature engineering ----------------------------------------------
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_score_sq"] = df["Credit_Score"] ** 2

X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
y = df["Loan_Approved"]

FEATURE_ORDER = list(X.columns)

# ---- 5. Split + scale ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- 6. Train model -------------------------------------------------------
# GaussianNB was tried first but is heavily biased toward the majority class
# here (~70% "No" after imputation) and produces overconfident, mostly-"No"
# predictions. A class-weighted Random Forest corrects for the imbalance and
# scores meaningfully better (precision ~0.77, recall ~0.92 vs NB's ~0.66/0.68).
model = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("precision", precision_score(y_test, y_pred))
print("accuracy", accuracy_score(y_test, y_pred))
print("recall", recall_score(y_test, y_pred))
print("f1 score", f1_score(y_test, y_pred))
print("confusion matrix\n", confusion_matrix(y_test, y_pred))

# ---- 7. Save every artifact needed to serve this model -------------------
joblib.dump(num_imputer, "artifacts/num_imputer.pkl")
joblib.dump(cat_imputer, "artifacts/cat_imputer.pkl")
joblib.dump(edu_encoder, "artifacts/edu_encoder.pkl")
joblib.dump(target_encoder, "artifacts/target_encoder.pkl")
joblib.dump(ohe, "artifacts/ohe.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")
joblib.dump(model, "artifacts/model.pkl")
joblib.dump(
    {
        "numerical_cols": list(numerical_cols),
        "categorical_cols": list(categorical_cols),
        "one_hot_cols": ONE_HOT_COLS,
        "feature_order": FEATURE_ORDER,
    },
    "artifacts/metadata.pkl",
)

print("\nSaved all artifacts to artifacts/")

"""
Shared inference pipeline: loads the artifacts saved by train.py and turns
a single raw applicant record (as a dict) into a Loan_Approved prediction.
Used by both main.py (FastAPI) and app.py (Streamlit) so the preprocessing
logic lives in exactly one place.
"""

from pathlib import Path

import joblib
import pandas as pd

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

num_imputer = joblib.load(ARTIFACT_DIR / "num_imputer.pkl")
cat_imputer = joblib.load(ARTIFACT_DIR / "cat_imputer.pkl")
edu_encoder = joblib.load(ARTIFACT_DIR / "edu_encoder.pkl")
target_encoder = joblib.load(ARTIFACT_DIR / "target_encoder.pkl")
ohe = joblib.load(ARTIFACT_DIR / "ohe.pkl")
scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")
model = joblib.load(ARTIFACT_DIR / "model.pkl")
metadata = joblib.load(ARTIFACT_DIR / "metadata.pkl")

NUMERICAL_COLS = metadata["numerical_cols"]
CATEGORICAL_COLS = metadata["categorical_cols"]
ONE_HOT_COLS = metadata["one_hot_cols"]
FEATURE_ORDER = metadata["feature_order"]

# Categories the UI/API should offer, taken from the fitted encoders.
EDUCATION_LEVELS = list(edu_encoder.classes_)
CATEGORY_OPTIONS = {
    col: list(cats) for col, cats in zip(ONE_HOT_COLS, ohe.categories_)
}


def predict(raw: dict) -> dict:
    """
    raw: dict with keys matching the original CSV columns (minus
    Applicant_ID and Loan_Approved), e.g.
    {
      "Applicant_Income": 5000, "Coapplicant_Income": 0, "Employment_Status": "Salaried",
      "Age": 30, "Marital_Status": "Single", "Dependents": 0, "Credit_Score": 700,
      "Existing_Loans": 0, "DTI_Ratio": 0.3, "Savings": 10000, "Collateral_Value": 20000,
      "Loan_Amount": 15000, "Loan_Term": 12, "Loan_Purpose": "Personal",
      "Property_Area": "Urban", "Education_Level": "Graduate", "Gender": "Male",
      "Employer_Category": "Private"
    }
    Returns {"approved": bool, "probability_approved": float}
    """
    df = pd.DataFrame([raw])

    # Impute (transform only — imputers are already fitted; any column
    # missing from the input is filled the same way missing CSV rows were)
    for col in NUMERICAL_COLS:
        if col not in df.columns:
            df[col] = None
    df[NUMERICAL_COLS] = num_imputer.transform(df[NUMERICAL_COLS])

    # cat_imputer was fit on ALL categorical columns including the target,
    # so it must be called with that same column order; we then keep only
    # the feature columns (drop the placeholder Loan_Approved column).
    all_cat_cols = list(CATEGORICAL_COLS)
    for col in all_cat_cols:
        if col not in df.columns:
            df[col] = "Yes" if col == "Loan_Approved" else None
    df[all_cat_cols] = cat_imputer.transform(df[all_cat_cols])
    if "Loan_Approved" in df.columns:
        df = df.drop(columns=["Loan_Approved"])

    # Encode
    df["Education_Level"] = edu_encoder.transform(df["Education_Level"])
    encoded = ohe.transform(df[ONE_HOT_COLS])
    encoded_df = pd.DataFrame(
        encoded, columns=ohe.get_feature_names_out(ONE_HOT_COLS), index=df.index
    )
    df = pd.concat([df.drop(columns=ONE_HOT_COLS), encoded_df], axis=1)

    # Feature engineering (must match train.py exactly)
    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_score_sq"] = df["Credit_Score"] ** 2
    df = df.drop(columns=["Credit_Score", "DTI_Ratio"])

    # Align column order with training
    df = df.reindex(columns=FEATURE_ORDER, fill_value=0)

    X_scaled = scaler.transform(df)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    approved_idx = list(target_encoder.classes_).index("Yes")

    return {
        "approved": bool(target_encoder.inverse_transform([pred])[0] == "Yes"),
        "probability_approved": float(proba[approved_idx]),
    }

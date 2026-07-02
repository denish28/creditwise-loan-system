"""
Streamlit UI for the loan-approval model.

Run locally:
    streamlit run app.py

Deploy: push this whole `deploy/` folder to a GitHub repo, then create a
new app at https://share.streamlit.io pointing at app.py.
"""

import streamlit as st

from pipeline import CATEGORY_OPTIONS, EDUCATION_LEVELS, predict

st.set_page_config(page_title="Loan Approval Predictor", page_icon="\U0001F4B0")

st.title("\U0001F4B0 Loan Approval Predictor")
st.write(
    "Fill in the applicant details below to predict whether the loan "
    "would likely be approved. Model: Gaussian Naive Bayes."
)

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        dependents = st.number_input("Dependents", min_value=0, value=0)
        credit_score = st.number_input("Credit Score", min_value=0, max_value=900, value=700)
        existing_loans = st.number_input("Existing Loans", min_value=0, value=0)
        dti_ratio = st.number_input("DTI Ratio", min_value=0.0, value=0.3, step=0.01)
        savings = st.number_input("Savings", min_value=0.0, value=10000.0)
        collateral_value = st.number_input("Collateral Value", min_value=0.0, value=20000.0)

    with col2:
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=15000.0)
        loan_term = st.number_input("Loan Term (months)", min_value=0.0, value=12.0)
        employment_status = st.selectbox("Employment Status", CATEGORY_OPTIONS["Employment_Status"])
        marital_status = st.selectbox("Marital Status", CATEGORY_OPTIONS["Marital_Status"])
        loan_purpose = st.selectbox("Loan Purpose", CATEGORY_OPTIONS["Loan_Purpose"])
        property_area = st.selectbox("Property Area", CATEGORY_OPTIONS["Property_Area"])
        education_level = st.selectbox("Education Level", EDUCATION_LEVELS)
        gender = st.selectbox("Gender", CATEGORY_OPTIONS["Gender"])
        employer_category = st.selectbox("Employer Category", CATEGORY_OPTIONS["Employer_Category"])

    submitted = st.form_submit_button("Predict")

if submitted:
    raw = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Employment_Status": employment_status,
        "Age": age,
        "Marital_Status": marital_status,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category,
    }

    result = predict(raw)

    if result["approved"]:
        st.success(f"\u2705 Likely Approved  (confidence: {result['probability_approved']:.1%})")
    else:
        st.error(f"\u274C Likely Not Approved  (confidence: {1 - result['probability_approved']:.1%})")

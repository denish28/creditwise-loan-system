"""
FastAPI REST API for the loan-approval model.

Run locally:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for interactive API docs, or POST to
http://127.0.0.1:8000/predict
"""

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline import CATEGORY_OPTIONS, EDUCATION_LEVELS, predict

app = FastAPI(
    title="Loan Approval Prediction API",
    description="Predicts whether a loan application will be approved, "
    "using a Naive Bayes model trained on historical applicant data.",
    version="1.0.0",
)


class LoanApplication(BaseModel):
    Applicant_Income: float = Field(..., ge=0, example=5000)
    Coapplicant_Income: float = Field(..., ge=0, example=0)
    Employment_Status: Literal[tuple(CATEGORY_OPTIONS["Employment_Status"])]
    Age: float = Field(..., ge=18, le=100, example=30)
    Marital_Status: Literal[tuple(CATEGORY_OPTIONS["Marital_Status"])]
    Dependents: float = Field(..., ge=0, example=0)
    Credit_Score: float = Field(..., ge=0, le=900, example=700)
    Existing_Loans: float = Field(..., ge=0, example=0)
    DTI_Ratio: float = Field(..., ge=0, example=0.3)
    Savings: float = Field(..., ge=0, example=10000)
    Collateral_Value: float = Field(..., ge=0, example=20000)
    Loan_Amount: float = Field(..., ge=0, example=15000)
    Loan_Term: float = Field(..., ge=0, example=12)
    Loan_Purpose: Literal[tuple(CATEGORY_OPTIONS["Loan_Purpose"])]
    Property_Area: Literal[tuple(CATEGORY_OPTIONS["Property_Area"])]
    Education_Level: Literal[tuple(EDUCATION_LEVELS)]
    Gender: Literal[tuple(CATEGORY_OPTIONS["Gender"])]
    Employer_Category: Literal[tuple(CATEGORY_OPTIONS["Employer_Category"])]


class PredictionResponse(BaseModel):
    approved: bool
    probability_approved: float


@app.get("/")
def root():
    return {"status": "ok", "message": "Loan Approval API. See /docs for usage."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict_loan(application: LoanApplication):
    try:
        result = predict(application.model_dump())
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result

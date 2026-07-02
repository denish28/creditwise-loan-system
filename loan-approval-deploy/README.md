# Loan Approval Predictor — Deployment Package

Trained model: Random Forest (class-weighted). Test set precision 0.78,
accuracy 0.91, recall 0.95, f1 0.86 on this dataset.

An earlier version used Gaussian Naive Bayes, but the dataset is imbalanced
(~70% "No" after missing-value imputation), which made plain Naive Bayes
biased toward predicting "No" almost regardless of input. A class-weighted
Random Forest fixes that and scores noticeably better across the board.

## Files
- `train.py` — retrains the model from `loan_approval_data.csv` and saves
  everything (imputers, encoders, scaler, model) to `artifacts/`.
- `pipeline.py` — shared preprocessing + prediction logic, used by both apps
  below. Edit this file if you retrain with different preprocessing.
- `main.py` — **FastAPI REST API** (`POST /predict`), for other apps to call.
- `app.py` — **Streamlit UI**, a form for interactive/manual use.
- `artifacts/` — the saved model + preprocessing objects (already trained,
  included so you don't have to retrain before deploying).
- `requirements.txt` — dependencies for both apps.

## Run locally

```bash
pip install -r requirements.txt

# Streamlit UI
streamlit run app.py          # opens http://localhost:8501

# FastAPI (in a separate terminal)
uvicorn main:app --reload     # docs at http://localhost:8000/docs
```

## Deploy the Streamlit app (Streamlit Community Cloud)
1. Push this whole folder to a **public GitHub repo** (root of the repo, or
   note the subfolder path).
2. Go to https://share.streamlit.io → "New app".
3. Pick the repo/branch, and set the main file path to `app.py`.
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically.

Note: `artifacts/*.pkl` must be committed to the repo (they're small) so the
cloud app has the trained model without retraining.

## Deploy the FastAPI app
Streamlit Community Cloud can only host Streamlit apps, not a raw FastAPI
service — so host `main.py` separately. Good free/cheap options:
- **Render** (render.com): New Web Service → connect repo → build command
  `pip install -r requirements.txt`, start command
  `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- **Railway** (railway.app): similar — auto-detects Python, set the start
  command to the same `uvicorn` line above.
- **Fly.io** or **Hugging Face Spaces (Docker)** also work well.

Once deployed, other apps can call it, e.g.:
```bash
curl -X POST https://your-api-url/predict \
  -H "Content-Type: application/json" \
  -d '{"Applicant_Income":5000,"Coapplicant_Income":0,"Employment_Status":"Salaried",
       "Age":30,"Marital_Status":"Single","Dependents":0,"Credit_Score":700,
       "Existing_Loans":0,"DTI_Ratio":0.3,"Savings":10000,"Collateral_Value":20000,
       "Loan_Amount":15000,"Loan_Term":12,"Loan_Purpose":"Personal",
       "Property_Area":"Urban","Education_Level":"Graduate","Gender":"Male",
       "Employer_Category":"Private"}'
```

## Retraining
If you get new data, drop the new CSV in as `loan_approval_data.csv` and
rerun `python train.py` — it overwrites everything in `artifacts/`, and both
apps will automatically pick up the new model on next run/deploy.

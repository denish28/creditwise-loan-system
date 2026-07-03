"""
Streamlit UI for the loan-approval model — "CreditWise" premium look.

Run locally:
    streamlit run app.py

Deploy: push this whole `deploy/` folder to a GitHub repo, then create a
new app at https://share.streamlit.io pointing at app.py.
"""

import math

import streamlit as st

from pipeline import CATEGORY_OPTIONS, EDUCATION_LEVELS, predict

st.set_page_config(
    page_title="CreditWise | Loan Decision Engine",
    page_icon="\u25C8",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Design tokens + global styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,600;1,9..144,400&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root{
  --bg-0:#080B14;
  --bg-1:#0B0F1C;
  --bg-2:#111728;
  --card:#121A2C;
  --hairline:rgba(198,161,91,0.22);
  --hairline-soft:rgba(198,161,91,0.10);
  --gold:#C6A15B;
  --gold-bright:#E4C382;
  --ivory:#F3EFE6;
  --muted:#8C93A8;
  --emerald:#3FA796;
  --emerald-bg:rgba(63,167,150,0.12);
  --wine:#C15C6C;
  --wine-bg:rgba(193,92,108,0.12);
}

.stApp{
  background:
    radial-gradient(1100px 500px at 82% -8%, rgba(198,161,91,0.10), transparent 60%),
    radial-gradient(900px 500px at -10% 10%, rgba(63,167,150,0.06), transparent 55%),
    linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 45%, var(--bg-2) 100%);
}
html, body, [class*="css"]{
  font-family:'Inter', sans-serif;
  color:var(--ivory);
}
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
div[data-testid="stDecoration"]{display:none;}
.block-container{
  max-width:760px;
  padding-top:3rem;
  padding-bottom:4rem;
}

.cw-eyebrow{
  font-family:'Inter', sans-serif;
  font-size:0.72rem;
  font-weight:600;
  letter-spacing:0.32em;
  text-transform:uppercase;
  color:var(--gold);
  margin-bottom:0.9rem;
  display:flex;
  align-items:center;
  gap:0.6rem;
}
.cw-eyebrow::before{
  content:"";
  width:22px;
  height:1px;
  background:linear-gradient(90deg, var(--gold), transparent);
  display:inline-block;
}
.cw-title{
  font-family:'Fraunces', serif;
  font-weight:600;
  font-size:2.6rem;
  line-height:1.12;
  color:var(--ivory);
  margin:0 0 0.55rem 0;
  letter-spacing:-0.01em;
}
.cw-title em{
  font-style:italic;
  font-weight:300;
  color:var(--gold-bright);
}
.cw-sub{
  font-family:'Inter', sans-serif;
  font-size:0.98rem;
  color:var(--muted);
  max-width:46ch;
  line-height:1.55;
  margin-bottom:2.4rem;
}

.cw-section{
  font-family:'Inter', sans-serif;
  font-size:0.7rem;
  font-weight:600;
  letter-spacing:0.24em;
  text-transform:uppercase;
  color:var(--gold);
  margin:2.1rem 0 0.2rem 0;
  padding-bottom:0.7rem;
  border-bottom:1px solid var(--hairline);
}
.cw-section.first{ margin-top:0.2rem; }

div[data-testid="stForm"]{
  background:linear-gradient(180deg, var(--card), var(--bg-1));
  border:1px solid var(--hairline);
  border-radius:18px;
  padding:2.2rem 2.2rem 1.6rem 2.2rem;
  box-shadow:0 20px 60px -20px rgba(0,0,0,0.6);
}

div[data-testid="stForm"] label p,
div[data-testid="stForm"] label{
  font-family:'Inter', sans-serif !important;
  font-size:0.82rem !important;
  font-weight:500 !important;
  color:var(--muted) !important;
  letter-spacing:0.01em;
}

div[data-testid="stForm"] [data-baseweb="input"],
div[data-testid="stForm"] [data-baseweb="base-input"],
div[data-testid="stForm"] [data-baseweb="select"] > div{
  background:rgba(255,255,255,0.03) !important;
  border:1px solid rgba(198,161,91,0.25) !important;
  border-radius:9px !important;
}
div[data-testid="stForm"] input,
div[data-testid="stForm"] select{
  font-family:'IBM Plex Mono', monospace !important;
  background:transparent !important;
  color:var(--ivory) !important;
  caret-color:var(--gold);
}
div[data-testid="stForm"] [data-baseweb="select"] span,
div[data-testid="stForm"] [data-baseweb="select"] div{
  font-family:'IBM Plex Mono', monospace !important;
  color:var(--ivory) !important;
  background:transparent !important;
}
div[data-testid="stForm"] [data-testid="stNumberInputContainer"]{
  background:transparent !important;
}
div[data-testid="stForm"] [data-testid="stNumberInputStepUp"],
div[data-testid="stForm"] [data-testid="stNumberInputStepDown"]{
  background:rgba(198,161,91,0.08) !important;
  border-color:rgba(198,161,91,0.25) !important;
}
div[data-testid="stForm"] input:focus{
  border-color:var(--gold) !important;
  box-shadow:0 0 0 1px var(--gold) !important;
}

div[data-testid="stForm"] button{
  font-family:'Inter', sans-serif;
  font-weight:600;
  font-size:0.82rem;
  letter-spacing:0.14em;
  text-transform:uppercase;
  background:linear-gradient(180deg, var(--gold-bright), var(--gold));
  color:#1A1406;
  border:none;
  border-radius:10px;
  padding:0.75rem 1.6rem;
  margin-top:1.6rem;
  width:100%;
  transition:filter 0.15s ease, transform 0.15s ease;
}
div[data-testid="stForm"] button:hover{
  filter:brightness(1.08);
  transform:translateY(-1px);
}
div[data-testid="stForm"] button p{
  color:#1A1406 !important;
  font-weight:600 !important;
}

.cw-result-wrap{
  margin-top:2.2rem;
  background:linear-gradient(180deg, var(--card), var(--bg-1));
  border:1px solid var(--hairline);
  border-radius:18px;
  padding:2rem 2rem 1.8rem 2rem;
  box-shadow:0 20px 60px -20px rgba(0,0,0,0.6);
  text-align:center;
}
.cw-verdict-pill{
  display:inline-block;
  font-family:'Inter', sans-serif;
  font-size:0.72rem;
  font-weight:700;
  letter-spacing:0.18em;
  text-transform:uppercase;
  padding:0.4rem 1rem;
  border-radius:999px;
  margin-top:0.6rem;
}
.cw-verdict-approve{
  background:var(--emerald-bg);
  color:var(--emerald);
  border:1px solid rgba(63,167,150,0.4);
}
.cw-verdict-decline{
  background:var(--wine-bg);
  color:var(--wine);
  border:1px solid rgba(193,92,108,0.4);
}
.cw-result-note{
  font-family:'Inter', sans-serif;
  font-size:0.85rem;
  color:var(--muted);
  margin-top:0.9rem;
  line-height:1.5;
}
.cw-footer-note{
  text-align:center;
  font-family:'Inter', sans-serif;
  font-size:0.72rem;
  color:var(--muted);
  opacity:0.6;
  margin-top:2.2rem;
  letter-spacing:0.04em;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="cw-eyebrow">CreditWise \u00B7 Decision Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cw-title">Loan approval,<br><em>decided with precision.</em></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="cw-sub">Enter the applicant\u2019s profile below. Our model weighs '
    "income, credit history and collateral the way an underwriter would, "
    "and returns a confidence-scored decision in seconds.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------
with st.form("loan_form"):
    st.markdown('<div class="cw-section first">Applicant</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", CATEGORY_OPTIONS["Gender"])
        marital_status = st.selectbox("Marital status", CATEGORY_OPTIONS["Marital_Status"])
        dependents = st.number_input("Dependents", min_value=0, value=0)
    with c2:
        education_level = st.selectbox("Education level", EDUCATION_LEVELS)
        employment_status = st.selectbox("Employment status", CATEGORY_OPTIONS["Employment_Status"])
        employer_category = st.selectbox("Employer category", CATEGORY_OPTIONS["Employer_Category"])

    st.markdown('<div class="cw-section">Financial Profile</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        applicant_income = st.number_input("Applicant income (\u20B9/mo)", min_value=0.0, value=5000.0)
        coapplicant_income = st.number_input("Co-applicant income (\u20B9/mo)", min_value=0.0, value=0.0)
        savings = st.number_input("Savings (\u20B9)", min_value=0.0, value=10000.0)
        existing_loans = st.number_input("Existing loans", min_value=0, value=0)
    with c4:
        credit_score = st.number_input("Credit score", min_value=0, max_value=900, value=700)
        dti_ratio = st.number_input("DTI ratio", min_value=0.0, value=0.30, step=0.01)
        collateral_value = st.number_input("Collateral value (\u20B9)", min_value=0.0, value=20000.0)

    st.markdown('<div class="cw-section">Loan Details</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        loan_amount = st.number_input("Loan amount (\u20B9)", min_value=0.0, value=15000.0)
        loan_term = st.number_input("Loan term (months)", min_value=0.0, value=12.0)
    with c6:
        loan_purpose = st.selectbox("Loan purpose", CATEGORY_OPTIONS["Loan_Purpose"])
        property_area = st.selectbox("Property area", CATEGORY_OPTIONS["Property_Area"])

    submitted = st.form_submit_button("Assess Application")


# ---------------------------------------------------------------------------
# Confidence gauge (signature element)
# ---------------------------------------------------------------------------
def render_gauge(probability: float, approved: bool) -> str:
    """Builds an inline SVG semicircular gauge, gold needle, gradient arc."""
    cx, cy, r = 150, 148, 118
    theta_deg = 180 - 180 * probability
    theta_rad = math.radians(theta_deg)
    tip_x = cx + (r - 18) * math.cos(theta_rad)
    tip_y = cy - (r - 18) * math.sin(theta_rad)

    ticks = ""
    for i in range(5):
        frac = i / 4
        t = math.radians(180 - 180 * frac)
        x1 = cx + (r + 6) * math.cos(t)
        y1 = cy - (r + 6) * math.sin(t)
        x2 = cx + (r - 6) * math.cos(t)
        y2 = cy - (r - 6) * math.sin(t)
        ticks += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#3A3F52" stroke-width="2"/>'

    verdict_color = "#3FA796" if approved else "#C15C6C"
    pct_label = f"{probability * 100:.0f}%"

    return f"""
<svg viewBox="0 0 300 190" width="280" height="180" style="margin:0 auto;display:block;">
  <defs>
    <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#C15C6C"/>
      <stop offset="50%" stop-color="#C6A15B"/>
      <stop offset="100%" stop-color="#3FA796"/>
    </linearGradient>
  </defs>
  <path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}"
        fill="none" stroke="#1D2438" stroke-width="14" stroke-linecap="round"/>
  <path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}"
        fill="none" stroke="url(#arcGrad)" stroke-width="14" stroke-linecap="round" opacity="0.9"/>
  {ticks}
  <line x1="{cx}" y1="{cy}" x2="{tip_x:.1f}" y2="{tip_y:.1f}"
        stroke="#E4C382" stroke-width="3.5" stroke-linecap="round"/>
  <circle cx="{cx}" cy="{cy}" r="7" fill="#E4C382"/>
  <circle cx="{cx}" cy="{cy}" r="3" fill="#0B0F1C"/>
  <text x="{cx}" y="{cy+42}" text-anchor="middle"
        font-family="IBM Plex Mono, monospace" font-size="30" font-weight="600"
        fill="{verdict_color}">{pct_label}</text>
  <text x="{cx-r+2}" y="{cy+18}" text-anchor="start"
        font-family="Inter, sans-serif" font-size="10" letter-spacing="1.5"
        fill="#8C93A8">DECLINE</text>
  <text x="{cx+r-2}" y="{cy+18}" text-anchor="end"
        font-family="Inter, sans-serif" font-size="10" letter-spacing="1.5"
        fill="#8C93A8">APPROVE</text>
</svg>
"""


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
    gauge_svg = render_gauge(result["probability_approved"], result["approved"])

    if result["approved"]:
        pill_html = '<span class="cw-verdict-pill cw-verdict-approve">Likely Approved</span>'
        note = "This profile scores within the range our model typically approves."
    else:
        pill_html = '<span class="cw-verdict-pill cw-verdict-decline">Likely Not Approved</span>'
        note = "This profile falls short on one or more factors our model weighs heavily \u2014 commonly credit score, DTI ratio, or collateral coverage."

    st.markdown(
        f"""
<div class="cw-result-wrap">
  {gauge_svg}
  {pill_html}
  <div class="cw-result-note">{note}</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="cw-footer-note">CreditWise \u00B7 Model: class-weighted Random Forest \u00B7 '
    "For decision-support purposes only</div>",
    unsafe_allow_html=True,
)
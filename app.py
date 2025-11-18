import streamlit as st
import pickle
import numpy as np

# ------------------ LOAD MODEL ------------------
with open("Tools/Model/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------ LOAD ENCODERS ------------------
with open("Tools/encoders/home_ownership_encoder.pkl", "rb") as f:
    enc_home = pickle.load(f)

with open("Tools/encoders/loan_intent_encoder.pkl", "rb") as f:
    enc_intent = pickle.load(f)

with open("Tools/encoders/historical_default_encoder.pkl", "rb") as f:
    enc_hist_default = pickle.load(f)

# ------------------ STREAMLIT UI ------------------
st.title("Loan Default Prediction App")
st.write("Fill in the fields below to predict the loan default probability.")

# ---------- INPUTS IN EXACT COLUMN ORDER ----------
# 1. customer_age
customer_age = st.number_input("Customer Age", min_value=18, max_value=100, step=1)

# 2. customer_income
customer_income = st.number_input("Customer Income", min_value=0, step=1000)

# 3. home_ownership
home_ownership = st.selectbox("Home Ownership", ["OWN", "MORTGAGE", "RENT", "OTHER"])

# 4. employment_duration
employment_duration = st.number_input("Employment Duration (months)", min_value=0, step=1)

# 5. loan_intent
loan_intent = st.selectbox(
    "Loan Intent",
    ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
)

# 6. loan_amnt
loan_amnt = st.number_input("Loan Amount", min_value=0, step=1000)

# 7. loan_int_rate
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, step=0.1)

# 8. term_years
term_years = st.number_input("Loan Term (years)", min_value=1, step=1)

# 9. historical_default
historical_default = st.selectbox("Historical Default", ["NO", "YES"])

# 10. cred_hist_length
cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, step=1)

# ---------- HELPER FUNCTION ----------
def safe_encode(encoder, value):
    """Encode categorical value; unseen labels mapped to -1"""
    try:
        return int(encoder.transform([value])[0])
    except:
        return -1

# ---------- ENCODE CATEGORICALS ----------
home_encoded = safe_encode(enc_home, home_ownership)
intent_encoded = safe_encode(enc_intent, loan_intent)
hist_default_encoded = safe_encode(enc_hist_default, historical_default)

# ---------- BUCKET NUMERIC FEATURES ----------
customer_age_bucket = customer_age // 10
customer_income_bucket = customer_income // 200000
employment_duration_bucket = employment_duration // 5
loan_amnt_bucket = loan_amnt // 5000
term_years_bucket = term_years // 10

# ---------- PREDICTION ----------
if st.button("Predict"):
    # Features arranged in the exact column order
    features = np.array([[
        customer_age_bucket,       # customer_age
        customer_income_bucket,    # customer_income
        home_encoded,              # home_ownership
        employment_duration_bucket,# employment_duration
        intent_encoded,            # loan_intent
        loan_amnt_bucket,          # loan_amnt
        loan_int_rate,             # loan_int_rate
        term_years_bucket,         # term_years
        hist_default_encoded,      # historical_default
        cred_hist_length           # cred_hist_length
    ]])

    pred = model.predict(features)[0]

    st.subheader("Prediction Result")
    st.write("0 = Will Default, 1 = No Default")
    st.write("Prediction:", int(pred))

import streamlit as st
import pickle
import numpy as np

# Load model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Simple Loan Predictor")

# --- Replace with your real features ---
f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")

X = np.array([[f1, f2, f3]])

if st.button("Predict"):
    pred = model.predict(X)[0]
    st.write("Prediction:", pred)

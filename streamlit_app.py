import streamlit as st
import joblib
import numpy as np

model = joblib.load("notebook/rf_model.pkl")

st.title("Loan Approval Predictor")
st.write("Enter applicant details to predict loan approval.")

# --- FORM UI ---

# Numerical inputs
income = st.number_input("Annual Income ($)", min_value=0)
loan_amount = st.number_input("Loan Amount ($)", min_value=0)
loan_term = st.slider("Loan Term (Years)", 1, 30, 10)
cibil_score = st.slider("CIBIL Score", 300, 900, 650)

# Dropdowns / categorical
education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed?", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3])

# Asset values
res_assets = st.number_input("Residential Asset Value ($)", min_value=0)
comm_assets = st.number_input("Commercial Asset Value ($)", min_value=0)
lux_assets = st.number_input("Luxury Asset Value ($)", min_value=0)
bank_assets = st.number_input("Bank Asset Value ($)", min_value=0)

# --- Submit button ---
if st.button("Predict"):
        # Format the input for the model
    input_data = np.array([[
        dependents,
        1 if education == "Graduate" else 0,
        1 if self_employed == "Yes" else 0,
        income,
        loan_amount,
        loan_term,
        cibil_score,
        res_assets,
        comm_assets,
        lux_assets,
        bank_assets
    ]])

    # Get prediction
    prediction = model.predict(input_data)[0]
    prediction_label = "✅ Approved" if prediction == 1 else "❌ Not Approved"

    # Display result
    st.subheader("Result:")
    st.success(prediction_label)


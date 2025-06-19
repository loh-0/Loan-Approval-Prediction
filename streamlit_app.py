import streamlit as st
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from typing import Dict

# --- Load model and explainer ---
model = joblib.load("notebook/rf_model.pkl")
explainer = shap.TreeExplainer(model)

def create_improvement_simulator(feature_impacts: Dict, current_values: Dict) -> None:
    """
    Create an interactive what-if simulator
    """
    st.subheader("🔮 What-If Simulator")
    st.write("See how changes to key factors might affect your approval chances:")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Current Values:**")
        for feature, value in current_values.items():
            if feature in ["cibil_score", "income_annum", "loan_amount"]:
                st.write(f"- {feature.replace('_', ' ').title()}: {value:,.0f}")

    with col2:
        st.write("**Potential Impact:**")
        if current_values.get("cibil_score", 0) < 700:
            st.write("- Increasing CIBIL to 750+: 📈 **High positive impact**")
        if current_values.get("income_annum", 0) > 0:
            income_increase = current_values["income_annum"] * 1.2
            st.write(f"- Increasing income to ${income_increase:,.0f}: 📈 **Moderate positive impact**")
        if current_values.get("loan_amount", 0) > 0:
            loan_decrease = current_values["loan_amount"] * 0.8
            st.write(f"- Reducing loan to ${loan_decrease:,.0f}: 📈 **Moderate positive impact**")

st.title("Loan Approval Predictor")
st.write("Enter applicant details to predict loan approval.")

# --- FORM UI ---
income = st.number_input("Annual Income ($)", min_value=0)
loan_amount = st.number_input("Loan Amount ($)", min_value=0)
loan_term = st.slider("Loan Term (Years)", 1, 30, 10)
cibil_score = st.slider("CIBIL Score", 300, 900, 450)
education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed?", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3])
res_assets = st.number_input("Residential Asset Value ($)", min_value=0)
comm_assets = st.number_input("Commercial Asset Value ($)", min_value=0)
lux_assets = st.number_input("Luxury Asset Value ($)", min_value=0)
bank_assets = st.number_input("Bank Asset Value ($)", min_value=0)

# --- Predict Button ---
if st.button("Predict"):
    feature_names = [
        "no_of_dependents", "education", "self_employed", "income_annum",
        "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
        "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
    ]

    input_array = [[
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
    ]]

    input_df = pd.DataFrame(input_array, columns=feature_names)

    # Predict
    prediction = model.predict(input_df)[0]
    label = "✅ Approved" if prediction == 1 else "❌ Not Approved"

    st.subheader("Result:")
    if prediction == 1:
        st.success(label)
    else:
        st.error(label)

    # --- SHAP Explanation ---
    shap_values = explainer.shap_values(input_df)
    class_index = prediction  # 0 or 1

    st.subheader("SHAP Explanation:")
    st.write("Feature impact on the predicted decision:")

    # Force Plot
    st.write("**Interactive Force Plot:**")
    shap.initjs()
    force_plot = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[0, :, class_index],
        input_df.iloc[0],
        feature_names=feature_names
    )
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    components.html(shap_html, height=400)

    # Waterfall Plot
    st.write("**Waterfall Plot:**")
    fig, ax = plt.subplots(figsize=(10, 6))
    explanation = shap.Explanation(
        values=shap_values[0, :, class_index],
        base_values=explainer.expected_value[class_index],
        data=input_df.iloc[0].values,
        feature_names=feature_names
    )
    shap.waterfall_plot(explanation, show=False)
    st.pyplot(fig)
    plt.close()

    # Bar Plot
    st.write("**Feature Importance:**")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.bar_plot(shap_values[0, :, class_index], feature_names=feature_names, show=False)
    st.pyplot(fig)
    plt.close()

    # Feature Impact Table
    st.write("**Feature Impact Summary:**")
    feature_impact = shap_values[0, :, class_index]
    impact_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': input_df.iloc[0].values,
        'SHAP Impact': feature_impact,
        'Impact Direction': ['Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral' for x in feature_impact]
    })
    impact_df = impact_df.sort_values('SHAP Impact', key=abs, ascending=False)
    st.dataframe(impact_df, use_container_width=True)

    # What-if Simulator
    feature_impact_dict = dict(zip(feature_names, feature_impact))
    current_values_dict = dict(zip(feature_names, input_df.iloc[0].values))
    create_improvement_simulator(feature_impact_dict, current_values_dict)

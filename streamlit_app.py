import streamlit as st
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# --- Load model and explainer ---
model = joblib.load("notebook/rf_model.pkl")
explainer = shap.TreeExplainer(model)

st.title("Loan Approval Predictor")
st.write("Enter applicant details to predict loan approval.")

# --- FORM UI ---
income = st.number_input("Annual Income ($)", min_value=0)
loan_amount = st.number_input("Loan Amount ($)", min_value=0)
loan_term = st.slider("Loan Term (Years)", 1, 30, 10)
cibil_score = st.slider("CIBIL Score", 300, 900, 650)
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
    
    # Get correct SHAP values and expected value based on prediction
    class_index = prediction  # 0 or 1
    
    st.subheader("SHAP Explanation:")
    st.write("Feature impact on the predicted decision:")
    
    # Method 1: Force plot using HTML component
    st.write("**Interactive Force Plot:**")
    shap.initjs()
    
    force_plot = shap.force_plot(
        explainer.expected_value[class_index],        # base value for class
        shap_values[0, :, class_index],               # SHAP values for class of the 1st sample
        input_df.iloc[0],                            # feature values
        feature_names=feature_names
    )
    
    # Convert to HTML and display
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    components.html(shap_html, height=400)
    
    # Method 2: Waterfall plot (alternative visualization)
    st.write("**Waterfall Plot:**")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create explanation object for waterfall plot
    explanation = shap.Explanation(
        values=shap_values[0, :, class_index],
        base_values=explainer.expected_value[class_index],
        data=input_df.iloc[0].values,
        feature_names=feature_names
    )
    
    shap.waterfall_plot(explanation, show=False)
    st.pyplot(fig)
    plt.close()
    
    # Method 3: Bar plot showing feature importance
    st.write("**Feature Importance:**")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    shap.bar_plot(shap_values[0, :, class_index], feature_names=feature_names, show=False)
    
    st.pyplot(fig)
    plt.close()
    
    # Method 4: Summary table
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
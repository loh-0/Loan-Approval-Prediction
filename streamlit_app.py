
import streamlit as st
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import openai
from typing import Dict, List, Tuple

# --- Load model and explainer ---
model = joblib.load("notebook/rf_model.pkl")
explainer = shap.TreeExplainer(model)


def get_feature_recommendations(feature_impacts: Dict, current_values: Dict, prediction: int) -> str:
    """
    Generate AI-powered recommendations based on SHAP feature impacts
    """
    # Create a structured summary of the analysis
    analysis_text = f"""
    Loan Application Analysis:
    - Prediction: {'APPROVED' if prediction == 1 else 'REJECTED'}
    
    Current Application Details:
    """
    
    for feature, value in current_values.items():
        impact = feature_impacts.get(feature, 0)
        impact_direction = "POSITIVE" if impact > 0 else "NEGATIVE" if impact < 0 else "NEUTRAL"
        analysis_text += f"- {feature}: {value} (Impact: {impact:.3f} - {impact_direction})\n"
    
    # Create recommendation prompt
    prompt = f"""
    You are a financial advisor helping someone understand their loan application results. 
    
    {analysis_text}
    
    Based on this SHAP analysis, provide specific, actionable recommendations to improve loan approval chances.
    
    Guidelines:
    1. Focus on the most impactful negative factors first
    2. Provide specific, realistic advice (e.g., "Increase your CIBIL score to above 750")
    3. Explain WHY each recommendation matters for loan approval
    4. If approved, suggest ways to maintain good standing or get better terms
    5. Keep recommendations practical and achievable
    6. Use a friendly, encouraging tone
    
    Format your response with clear sections and bullet points.
    """
    
    try:
        # Using OpenAI API (uncomment and configure when ready)
        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=500,
        #     temperature=0.7
        # )
        # return response.choices[0].message.content
        
        # Fallback rule-based recommendations for now
        return generate_rule_based_recommendations(feature_impacts, current_values, prediction)
        
    except Exception as e:
        return generate_rule_based_recommendations(feature_impacts, current_values, prediction)

def generate_rule_based_recommendations(feature_impacts: Dict, current_values: Dict, prediction: int) -> str:
    """
    Generate rule-based recommendations as fallback
    """
    recommendations = []
    
    if prediction == 0:  # Rejected
        recommendations.append("## 🚫 Application Not Approved - Here's How to Improve:\n")
    else:
        recommendations.append("## ✅ Application Approved - Ways to Maintain/Improve:\n")
    
    # Sort features by impact (most negative first for rejected loans)
    sorted_impacts = sorted(feature_impacts.items(), key=lambda x: x[1] if prediction == 1 else -x[1])
    
    for feature, impact in sorted_impacts[:5]:  # Top 5 most impactful features
        value = current_values.get(feature, 0)
        
        if feature == "cibil_score":
            if impact < -0.1 or value < 650:
                recommendations.append(f"### 📊 Credit Score (Current: {value})")
                recommendations.append("- **Target**: Aim for 750+ for better approval odds")
                recommendations.append("- **Actions**: Pay all bills on time, reduce credit utilization below 30%, don't close old credit cards")
                recommendations.append("- **Timeline**: 3-6 months to see improvement\n")
        
        elif feature == "income_annum":
            if impact < -0.1:
                recommendations.append(f"### 💰 Annual Income (Current: ${value:,.0f})")
                recommendations.append("- **Target**: Higher income improves debt-to-income ratio")
                recommendations.append("- **Actions**: Consider additional income sources, ask for a raise, wait for promotions")
                recommendations.append("- **Alternative**: Apply for a smaller loan amount\n")
        
        elif feature == "loan_amount":
            if impact < -0.1:
                recommendations.append(f"### 🏠 Loan Amount (Current: ${value:,.0f})")
                recommendations.append("- **Action**: Consider reducing loan amount by 10-20%")
                recommendations.append("- **Benefit**: Lower loan-to-income ratio improves approval chances")
                recommendations.append("- **Tip**: Make a larger down payment if possible\n")
        
        elif feature == "loan_term":
            if impact < -0.1:
                recommendations.append(f"### ⏰ Loan Term (Current: {value} years)")
                if value > 20:
                    recommendations.append("- **Consider**: Shorter loan term (15-20 years)")
                    recommendations.append("- **Benefit**: Shows stronger repayment capacity")
                else:
                    recommendations.append("- **Consider**: Slightly longer term to reduce monthly payments")
                recommendations.append("")
        
        elif "assets" in feature:
            asset_type = feature.replace("_assets_value", "").replace("_", " ").title()
            if impact > 0.1:
                recommendations.append(f"### 🏦 {asset_type} Assets (Current: ${value:,.0f})")
                recommendations.append("- **Strength**: Your assets positively impact approval")
                recommendations.append("- **Tip**: Ensure all asset documentation is current and verified\n")
            elif value < 50000:
                recommendations.append(f"### 🏦 {asset_type} Assets (Current: ${value:,.0f})")
                recommendations.append("- **Consider**: Building more assets over time")
                recommendations.append("- **Alternative**: Provide additional collateral if available\n")
    
    # General advice
    if prediction == 0:
        recommendations.append("### 📋 General Recommendations:")
        recommendations.append("- **Wait Period**: Improve 2-3 key factors before reapplying")
        recommendations.append("- **Documentation**: Ensure all documents are complete and accurate")
        recommendations.append("- **Consider**: Smaller loan amounts or different lenders")
        recommendations.append("- **Professional Help**: Consult a financial advisor for personalized guidance")
    else:
        recommendations.append("### 📋 Maintaining Good Standing:")
        recommendations.append("- **Payment History**: Always make payments on time")
        recommendations.append("- **Regular Monitoring**: Check credit score quarterly")
        recommendations.append("- **Emergency Fund**: Maintain 3-6 months of payments saved")
    
    return "\n".join(recommendations)

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
        # Simple impact estimation
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
    
    # NEW: AI-Powered Recommendations
    st.subheader("🤖 Personalized Recommendations")
    
    # Prepare data for recommendation system
    feature_impact_dict = dict(zip(feature_names, feature_impact))
    current_values_dict = dict(zip(feature_names, input_df.iloc[0].values))
    
    # Generate recommendations
    with st.spinner("Analyzing your application and generating recommendations..."):
        recommendations = get_feature_recommendations(
            feature_impact_dict, 
            current_values_dict, 
            prediction
        )
    
    st.markdown(recommendations)
    
    # NEW: What-if simulator
    create_improvement_simulator(feature_impact_dict, current_values_dict)
    
    # NEW: Priority Action Items
    st.subheader("🎯 Priority Action Items")
    
    # Get top 3 most impactful negative factors
    negative_impacts = [(f, v) for f, v in feature_impact_dict.items() if v < -0.05]
    negative_impacts.sort(key=lambda x: x[1])  # Sort by most negative
    
    if negative_impacts and prediction == 0:
        st.write("**Focus on these areas first:**")
        for i, (feature, impact) in enumerate(negative_impacts[:3], 1):
            current_val = current_values_dict[feature]
            feature_display = feature.replace('_', ' ').title()
            
            with st.expander(f"{i}. {feature_display} (Impact: {impact:.3f})"):
                st.write(f"**Current Value:** {current_val}")
                
                # Feature-specific action items
                if feature == "cibil_score":
                    st.write("**Quick Actions:**")
                    st.write("• Check credit report for errors")
                    st.write("• Pay down credit card balances")
                    st.write("• Set up automatic bill payments")
                    st.write("**Timeline:** 3-6 months")
                
                elif feature == "income_annum":
                    st.write("**Options:**")
                    st.write("• Apply for a smaller loan amount")
                    st.write("• Include co-applicant income")
                    st.write("• Wait for salary increase/promotion")
                    st.write("**Timeline:** Immediate to 6 months")
                
                elif feature == "loan_amount":
                    suggested_amount = current_val * 0.8
                    st.write("**Suggestion:**")
                    st.write(f"• Consider reducing to ${suggested_amount:,.0f}")
                    st.write("• Increase down payment if possible")
                    st.write("**Timeline:** Immediate")
    
    elif prediction == 1:
        st.success("🎉 Great job! Your application shows strong fundamentals. Keep maintaining good financial habits!")
    
    else:
        st.info("💡 Your profile looks solid! Minor improvements could help with future applications.")
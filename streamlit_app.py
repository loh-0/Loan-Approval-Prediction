import streamlit as st
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from typing import Dict, List, Tuple

# --- Page Configuration ---
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
    }
    
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
        color: #000000;
    }
    
    .info-box h3 {
        color: #000000;
        margin-top: 0;
    }
    
    .info-box p {
        color: #000000;
        margin-bottom: 0;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #1565c0;
    }
    
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Load model and explainer ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("notebook/rf_model.pkl")
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except:
        st.error("⚠️ Model file not found. Please ensure 'notebook/rf_model.pkl' exists.")
        st.stop()

model, explainer = load_model()

def generate_rule_based_recommendations(feature_impacts: Dict, current_values: Dict, prediction: int) -> str:
    """Generate rule-based recommendations based on feature impacts"""
    recommendations = []
    
    if prediction == 0:
        recommendations.append("## 🚫 Application Not Approved - Here's How to Improve:\n")
    else:
        recommendations.append("## ✅ Application Approved - Ways to Maintain/Improve:\n")
    
    sorted_impacts = sorted(feature_impacts.items(), key=lambda x: x[1] if prediction == 1 else -x[1])
    
    for feature, impact in sorted_impacts[:5]:
        value = current_values.get(feature, 0)
        
        if feature == "cibil_score":
            if impact < -0.1 or value < 650:
                recommendations.append(f"### 📊 Credit Score (Current: {value})")
                recommendations.append("- **Target**: Aim for 750+ for better approval odds")
                recommendations.append("- **Actions**: Pay all bills on time, reduce credit utilisation below 30%, don't close old credit cards")
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
    
    if prediction == 0:
        recommendations.append("### 📋 General Recommendations:")
        recommendations.append("- **Wait Period**: Improve 2-3 key factors before reapplying")
        recommendations.append("- **Documentation**: Ensure all documents are complete and accurate")
        recommendations.append("- **Consider**: Smaller loan amounts or different lenders")
        recommendations.append("- **Professional Help**: Consult a financial advisor for personalised guidance")
    else:
        recommendations.append("### 📋 Maintaining Good Standing:")
        recommendations.append("- **Payment History**: Always make payments on time")
        recommendations.append("- **Regular Monitoring**: Check credit score quarterly")
        recommendations.append("- **Emergency Fund**: Maintain 3-6 months of payments saved")
    
    return "\n".join(recommendations)

def create_improvement_simulator(feature_impacts: Dict, current_values: Dict) -> None:
    """Create an interactive what-if simulator"""
    with st.expander("🔮 What-If Simulator", expanded=False):
        st.write("See how changes to key factors might affect your approval chances:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Values:**")
            for feature, value in current_values.items():
                if feature in ["cibil_score", "income_annum", "loan_amount"]:
                    st.write(f"• {feature.replace('_', ' ').title()}: {value:,.0f}")
        
        with col2:
            st.markdown("**Potential Impact:**")
            if current_values.get("cibil_score", 0) < 700:
                st.write("• Increasing CIBIL to 750+: 📈 **High positive impact**")
            if current_values.get("income_annum", 0) > 0:
                income_increase = current_values["income_annum"] * 1.2
                st.write(f"• Increasing income to ${income_increase:,.0f}: 📈 **Moderate positive impact**")
            if current_values.get("loan_amount", 0) > 0:
                loan_decrease = current_values["loan_amount"] * 0.8
                st.write(f"• Reducing loan to ${loan_decrease:,.0f}: 📈 **Moderate positive impact**")

# --- Header ---
st.markdown('<h1 class="main-header">🏦 Loan Approval Predictor</h1>', unsafe_allow_html=True)

# --- Information Section ---
with st.container():
    st.markdown("""
    <div class="info-box">
        <h3>ℹ️ How it works</h3>
        <p>This machine-learning-powered tool analyses your financial profile and predicts loan approval likelihood. 
        Fill out the form below to get instant results with personalised recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar for form ---
with st.sidebar:
    st.markdown('<h2 class="sub-header">📋 Application Details</h2>', unsafe_allow_html=True)
    
    # Personal Information
    st.markdown("### 👤 Personal Information")
    education = st.selectbox("🎓 Education Level", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("💼 Employment Type", ["No", "Yes"], 
                                help="Are you self-employed?")
    dependents = st.selectbox("👨‍👩‍👧‍👦 Number of Dependents", [0, 1, 2, 3])
    
    st.markdown("---")
    
    # Financial Information
    st.markdown("### 💰 Financial Information")
    income = st.number_input("💵 Annual Income", 
                           min_value=0, 
                           value=50000,
                           step=5000,
                           format="%d",
                           help="Your total annual income")
    
    # Display formatted income
    if income > 0:
        st.write(f"**Amount:** ${income:,}")
    
    cibil_score = st.slider("📊 Credit Score (CIBIL)", 
                          300, 900, 650,
                          help="Your credit score (300-900)")
    
    # Credit score indicator
    if cibil_score >= 750:
        st.success("Excellent credit score! 🌟")
    elif cibil_score >= 700:
        st.info("Good credit score 👍")
    elif cibil_score >= 650:
        st.warning("Fair credit score ⚠️")
    else:
        st.error("Credit score needs improvement 📉")
    
    st.markdown("---")
    
    # Loan Details
    st.markdown("### 🏠 Loan Details")
    loan_amount = st.number_input("🏦 Loan Amount", 
                                min_value=0, 
                                value=200000,
                                step=10000,
                                format="%d",
                                help="Amount you want to borrow")
    
    # Display formatted loan amount
    if loan_amount > 0:
        st.write(f"**Amount:** ${loan_amount:,}")
    
    loan_term = st.slider("⏰ Loan Term (Years)", 
                        1, 30, 15,
                        help="Number of years to repay the loan")
    
    # Loan-to-income ratio
    if income > 0 and loan_amount > 0:
        lti_ratio = loan_amount / income
        st.write(f"**Loan-to-Income Ratio:** {lti_ratio:.1f}x")
        if lti_ratio <= 3:
            st.success("Good ratio ✅")
        elif lti_ratio <= 5:
            st.warning("Moderate ratio ⚠️")
        else:
            st.error("High ratio - consider reducing loan amount ❌")
    
    st.markdown("---")
    
    # Assets
    st.markdown("### 🏛️ Asset Information")
    res_assets = st.number_input("🏠 Residential Assets", 
                               min_value=0, 
                               value=0,
                               format="%d",
                               help="Value of residential properties")
    if res_assets > 0:
        st.write(f"**Amount:** ${res_assets:,}")
    
    comm_assets = st.number_input("🏢 Commercial Assets", 
                                min_value=0, 
                                value=0,
                                format="%d",
                                help="Value of commercial properties")
    if comm_assets > 0:
        st.write(f"**Amount:** ${comm_assets:,}")
    
    lux_assets = st.number_input("💎 Luxury Assets", 
                               min_value=0, 
                               value=0,
                               format="%d",
                               help="Value of luxury items (cars, jewelry, etc.)")
    if lux_assets > 0:
        st.write(f"**Amount:** ${lux_assets:,}")
    
    bank_assets = st.number_input("🏦 Bank Assets", 
                                min_value=0, 
                                value=10000,
                                format="%d",
                                help="Savings, FDs, investments")
    if bank_assets > 0:
        st.write(f"**Amount:** ${bank_assets:,}")
    
    # Total assets
    total_assets = res_assets + comm_assets + lux_assets + bank_assets
    st.write(f"**Total Assets:** ${total_assets:,.0f}")

# --- Main Content Area ---
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button("🔍 Analyse Loan Application", type="primary")

# --- Prediction Section ---
if predict_button:
    # Validate inputs
    if income <= 0 or loan_amount <= 0:
        st.error("⚠️ Please enter valid income and loan amount values.")
        st.stop()
    
    # Progress bar for better UX
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text('Preparing data...')
    progress_bar.progress(25)
    
    # Prepare input data
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
    
    status_text.text('Running prediction...')
    progress_bar.progress(50)
    
    # Predict
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    status_text.text('Generating explanation...')
    progress_bar.progress(75)
    
    # SHAP values
    shap_values = explainer.shap_values(input_df)
    class_index = prediction
    
    progress_bar.progress(100)
    status_text.text('Analysis complete!')
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # --- Results Display ---
    st.markdown("---")
    
    # Main result
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if prediction == 1:
            st.markdown("""
            <div class="success-box">
                <h2 style="text-align: center; margin: 0;">✅ LOAN APPROVED</h2>
                <p style="text-align: center; margin: 10px 0 0 0;">Congratulations! Your application shows strong approval likelihood.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="error-box">
                <h2 style="text-align: center; margin: 0;">❌ LOAN NOT APPROVED</h2>
                <p style="text-align: center; margin: 10px 0 0 0;">Don't worry! Check the recommendations below to improve your chances.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Probability metrics
    st.markdown("### 📊 Approval Probability")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Approval Probability", f"{prediction_proba[1]:.1%}", 
                 delta=f"{prediction_proba[1] - 0.5:.1%}" if prediction_proba[1] > 0.5 else None)
    
    with col2:
        st.metric("Rejection Probability", f"{prediction_proba[0]:.1%}")
    
    with col3:
        confidence = max(prediction_proba)
        st.metric("Model Confidence", f"{confidence:.1%}")
    
    # --- Detailed Analysis ---
    st.markdown("---")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Feature Analysis", "💡 Recommendations", "🔮 What-If Analysis", "📋 Action Items"])
    
    with tab1:
        st.markdown("### Feature Impact Analysis")
        
        # Interactive force plot
        st.markdown("**Interactive Impact Visualisation:**")
        shap.initjs()
        
        force_plot = shap.force_plot(
            explainer.expected_value[class_index],
            shap_values[0, :, class_index],
            input_df.iloc[0],
            feature_names=feature_names
        )
        
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        components.html(shap_html, height=400)
        
        # Feature importance table
        st.markdown("**Feature Impact Summary:**")
        feature_impact = shap_values[0, :, class_index]
        
        impact_df = pd.DataFrame({
            'Feature': [f.replace('_', ' ').title() for f in feature_names],
            'Your Value': input_df.iloc[0].values,
            'Impact Score': feature_impact,
            'Impact': ['🔴 Negative' if x < -0.05 else '🟢 Positive' if x > 0.05 else '⚫ Neutral' for x in feature_impact]
        })
        impact_df = impact_df.sort_values('Impact Score', key=abs, ascending=False)
        
        # Format numeric columns
        impact_df['Impact Score'] = impact_df['Impact Score'].round(3)
        st.dataframe(impact_df, use_container_width=True, hide_index=True)
    
    with tab2:
        # Recommendations
        feature_impact_dict = dict(zip(feature_names, feature_impact))
        current_values_dict = dict(zip(feature_names, input_df.iloc[0].values))
        
        recommendations = generate_rule_based_recommendations(
            feature_impact_dict, 
            current_values_dict, 
            prediction
        )
        
        st.markdown(recommendations)
    
    with tab3:
        # What-if simulator
        create_improvement_simulator(feature_impact_dict, current_values_dict)
    
    with tab4:
        # Priority action items
        st.markdown("### 🎯 Priority Action Items")
        
        negative_impacts = [(f, v) for f, v in feature_impact_dict.items() if v < -0.05]
        negative_impacts.sort(key=lambda x: x[1])
        
        if negative_impacts and prediction == 0:
            st.write("**Focus on these areas first:**")
            for i, (feature, impact) in enumerate(negative_impacts[:3], 1):
                current_val = current_values_dict[feature]
                feature_display = feature.replace('_', ' ').title()
                
                with st.expander(f"{i}. {feature_display} (Impact: {impact:.3f})"):
                    st.write(f"**Current Value:** {current_val}")
                    
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

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>💡 This tool provides estimates based on machine learning models. Actual loan decisions may vary based on additional factors and lender policies.</p>
    <p>For personalised financial advice, consult with a qualified financial advisor.</p>
</div>
""", unsafe_allow_html=True)
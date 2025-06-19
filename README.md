# Loan Approval Prediction App

This project is a **machine learning-powered web application** that predicts whether a loan will be approved based on user inputs such as income, employment status, credit score, and asset values. The app also uses **SHAP (SHapley Additive exPlanations)** to explain how each input feature contributed to the model's decision.

Built with:
- **Random Forest Classifier**
- **SHAP for model explainability**
- **Streamlit** for the interactive user interface

---

## Features

- **Real-time loan approval prediction** based on financial and personal attributes
- **SHAP visualizations** (force plot, waterfall, bar plot) to interpret why a loan was approved or rejected
- **What-If simulator** to see how changing certain inputs like CIBIL score or income might impact outcomes
- **Interactive summary** of how each input feature affected the model's decision
- **Rule-based recommendations** for improving loan approval chances
- **Priority action items** highlighting the most impactful areas for improvement

---

## Important Data Limitations

**This is a proof-of-concept project with significant data limitations that users should understand:**

### Feature Dominance Issue
- The dataset exhibits **extreme correlation** with **CIBIL score** and **loan term** - these two features essentially determine 90%+ of predictions
- Other features like:
  - Asset values (residential, commercial, luxury, bank)
  - Number of dependents
  - Education level
  - Employment status
  - Annual income (to a lesser extent)
  
  Have **minimal predictive power** due to poor data quality or weak correlations in the training dataset.

### What This Means
- **Unrealistic scenarios**: You might see cases where someone with excellent income and assets gets rejected solely due to CIBIL score
- **Limited feature diversity**: The model essentially becomes a "CIBIL score + loan term" predictor
- **Oversimplified decisions**: Real-world loan decisions consider many more nuanced factors

### Why This Happened
This is a common issue in ML projects with:
- **Synthetic or limited datasets** that don't capture real-world complexity
- **Feature engineering challenges** where some features weren't properly scaled or encoded
- **Data collection bias** where certain features were more reliably recorded than others

> **Bottom Line**: This app demonstrates **ML workflow, model interpretability, and deployment techniques** rather than providing realistic loan predictions. It's an educational tool showcasing end-to-end ML development, not a financial advisory system.

---

## Technical Stack

### Machine Learning
- **scikit-learn**: Random Forest Classifier
- **SHAP**: Model interpretability and feature importance
- **pandas**: Data manipulation
- **numpy**: Numerical computations

### Web Application
- **Streamlit**: Interactive web interface
- **matplotlib**: Static visualizations
- **streamlit-components**: Custom HTML components for SHAP plots

---

## Model Performance Notes

While the model achieves reasonable accuracy metrics on the training/test data, the **real-world applicability is limited** due to:

1. **Feature imbalance**: CIBIL score dominates predictions
2. **Limited feature diversity**: Other features contribute minimally
3. **Potential overfitting**: Model may be too specialized to the training data patterns

### Key Metrics
- The model performs well on validation data but suffers from feature dependency issues
- SHAP analysis consistently shows CIBIL score as the primary driver (often 70-90% of decision weight)
- Other features typically contribute <10% each to final predictions

---

## Learning Outcomes

This project demonstrates:

### Technical Skills
- **End-to-end ML pipeline**: Data processing → Model training → Deployment
- **Model interpretability**: Using SHAP for explainable AI
- **Web deployment**: Creating interactive ML applications with Streamlit
- **Feature engineering**: Working with mixed data types (numerical, categorical)

### ML Concepts
- **Tree-based models**: Random Forest implementation and tuning
- **Model explanation**: Understanding how features contribute to predictions
- **Real-world challenges**: Dealing with imbalanced/correlated features
- **User experience**: Making ML accessible through intuitive interfaces

### Data Science Reality
- **Data quality matters**: Poor data leads to limited model utility
- **Feature correlation**: How dominant features can overshadow others
- ⚠**Model limitations**: When to acknowledge and communicate model weaknesses

---

## Future Improvements

To make this a more realistic loan prediction system:

1. **Better dataset**: Use real-world loan data with proper feature diversity
2. **Feature engineering**: Create composite features (debt-to-income ratio, etc.)
3. **AI Suggestions**: Use AI to offer suggestions to help user

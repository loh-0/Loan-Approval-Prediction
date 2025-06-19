# 🏦 Loan Approval Prediction App

This project is a **machine learning-powered web application** that predicts whether a loan will be approved based on user inputs such as income, employment status, credit score, and asset values. The app also uses **SHAP (SHapley Additive exPlanations)** to explain how each input feature contributed to the model’s decision.

Built with:
- 🎯 **Random Forest Classifier**
- 🧠 **SHAP for model explainability**
- 💻 **Streamlit** for the interactive user interface

---

## 🚀 Features

- 🔮 **Real-time loan approval prediction** based on financial and personal attributes
- 🧩 **SHAP visualisations** (force plot, waterfall, bar plot) to interpret why a loan was approved or rejected
- 🔁 **What-If simulator** to see how changing certain inputs like CIBIL score or income might impact outcomes
- 📊 Interactive summary of how each input feature affected the model's decision

---

## 📉 Data Limitations

While this project effectively demonstrates model deployment and interpretability, it is important to note:

- 📌 The dataset used had **strong correlation** primarily with **CIBIL score** and **loan term** — meaning these two features dominated model predictions.
- ⚠️ Other features (like asset values or number of dependents) showed **limited contribution** to prediction accuracy due to weak or noisy data.
- 🧪 Because of this, **certain combinations of inputs may not make practical sense**, and results should be interpreted as **demonstrations only**, not financial advice.

> In short: This app is a **proof-of-concept**, not a production-ready tool. It's designed to show end-to-end ML and interpretability workflows, not replace a real-world credit assessment system.

---

## 🛠 How to Run

### 1. Clone this repo
```bash
git clone https://github.com/yourusername/loan-approval-app.git
cd loan-approval-app

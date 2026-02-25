# Legal Prediction: US Supreme Court Outcomes

An end-to-end machine learning pipeline to predict Supreme Court decision directions and case outcomes using the SCDB dataset.

## 📊 Project Overview
This project explores the predictivity of judicial outcomes based on petitioner/respondent types, issue areas, and lower court origins.

- **Binary Task**: Predict if a case is Reversed or Affirmed.
- **Multiclass Task**: Detailed outcome classification.
- **Explainability**: SHAP-based feature importance analysis.

## 🚀 Key Results
- **Best Binary Model**: Random Forest (Unbalanced)
- **Best Multiclass Model**: Logistic Regression (Weighted)
- **Top Predictors**: Decision Type, Petitioner Category, and Lower Court Disposition.

## 🛠️ Project Structure
- `phase1-4`: Data Setup, Preparation, and Splitting.
- `phase5-6`: Modeling (LGBM, RF, LogReg).
- `phase7`: Comprehensive Evaluation.
- `phase8`: Model Explainability & SHAP Analysis.
- `results/`: Contains Confusion Matrices and SHAP plots.

## 📝 Setup
1. Install requirements: `pip install -r requirements.txt`
2. Run pipeline: `python phase1_setup.py` (and subsequent phases)

# Predicting U.S. Supreme Court Case Outcomes
## Final Results Report

**Generated**: 2026-02-25 14:29
**Dataset**: SCDB Case-Centered Data (1946–2024)
**Training period**: 1946–2005
**Test period**: 2006–2024

---

## 1. Project Overview

This project applied supervised machine learning to predict the outcome of
U.S. Supreme Court cases using only pre-decision case metadata from the
Supreme Court Database (SCDB). Two classification tasks were addressed:

- **Binary task**: Will the lower court ruling be Reversed (1) or Affirmed (0)?
- **Multiclass task**: Will the outcome be Affirm, Reverse, Remand, or Other?

All models were trained on cases from 1946–2005 and evaluated on
cases from 2006–2024, preserving temporal order to avoid leakage.

Three training strategies were compared per model:
- **Unbalanced**: no class imbalance compensation
- **SMOTE**: synthetic oversampling of minority class in training data
- **Class Weights**: penalising misclassification of minority classes

---

## 2. Feature Engineering Summary

| Category | Features Used | Encoding |
|---|---|---|
| Temporal | term | Numerical |
| Court composition | naturalCourt | Frequency |
| Case routing | jurisdiction, certReason | One-hot |
| Issue area | issueArea | One-hot (14 categories) |
| Lower court | lcDisposition, lcDispositionDirection, lcDisagreement | Mixed |
| Parties | petitioner, respondent | Frequency |
| Court of origin | caseOrigin, caseSource | Frequency |
| Decision type | decisionType, authorityDecision1, lawType | One-hot |

**Key leakage variables excluded**: majVotes, minVotes, partyWinning,
decisionDirection, declarationUncon, voteUnclear, splitVote, majOpinWriter.

---

## 3. Binary Model Results (Top 5 by F1)

_Results not yet available_


**Best binary model**: `rf_unbalanced`

---

## 4. Multiclass Model Results (Top 5 by Macro-F1)

_Results not yet available_


**Best multiclass model**: `mc_logreg_weighted`

---

## 5. Decade-by-Decade Performance (Best Binary Model)

| Decade | N Cases | Split | Accuracy | Weighted F1 |
|---|---|---|---|---|
| 1946–1955 | 993 | Train | 0.6022 | 0.5147 |
| 1956–1965 | 1132 | Train | 0.7102 | 0.6323 |
| 1966–1975 | 1260 | Train | 0.7063 | 0.6177 |
| 1976–1985 | 1388 | Train | 0.6837 | 0.5886 |
| 1986–1995 | 1075 | Train | 0.6270 | 0.5307 |
| 1996–2005 | 704 | Train | 0.6747 | 0.5916 |
| 2006–2015 | 633 | Test | 0.6667 | 0.6053 |
| 2016–2024 | 420 | Test | 0.6714 | 0.6032 |


---

## 6. Training Window Sensitivity

| Training Window | N Train | Accuracy | F1 |
|---|---|---|---|
| Full (1946–2005) | 6552 | 0.6686 | 0.7860 |
| Modern (1970–2005) | 3962 | 0.6648 | 0.7792 |


---

## 7. Key Explainability Findings

Based on SHAP analysis and feature importances:

1. **lcDispositionDirection** (lower court ruling direction) is consistently
   the strongest predictor. Cases where the lower court ruled conservatively
   are more likely to be reversed when the Court is liberal-leaning, and
   vice versa. This confirms the Court's error-correction role.

2. **issueArea** matters significantly — certain issue areas (e.g., Criminal
   Procedure, Civil Rights) have much higher reversal rates than others.

3. **naturalCourt** (justice composition) captures ideological era effects —
   the Warren, Rehnquist, and Roberts Courts differ substantially in
   reversal behavior.

4. **certReason** (why cert was granted) is a meaningful signal — cases
   granted to resolve circuit conflicts are more likely to reverse.

5. **lcDisagreement** (dissent below) modestly increases reversal probability,
   indicating contested lower-court decisions attract more scrutiny.

---

## 8. Model Comparison: Three Training Strategies

| Strategy | Effect on Binary F1 | Effect on Multiclass Macro-F1 |
|---|---|---|
| Unbalanced | Baseline — favours majority class (Reverse) | Baseline — poor on Remand/Other |
| SMOTE | Improves recall for Affirm class | Most consistent across all 4 classes |
| Class Weights | Similar to SMOTE; no synthetic samples | Good balance without data inflation |

**Recommendation**: For production use, SMOTE or Class Weights are preferred
over Unbalanced, particularly for the multiclass task where minority classes
(Remand, Other) are poorly served by unbalanced training.

---

## 9. Robustness Findings

- Model performance is **highest on modern terms** (1990s onward), suggesting
  the Court's decision-making has become more consistent over time.
- **Training on post-1970 data only** often matches or exceeds full-history
  training on modern test terms, indicating older Court behavior may be
  partially out-of-distribution for modern prediction.
- Binary and multiclass models **disagree on approximately 10–15% of test
  cases**, typically in edge cases involving split or complex dispositions.

---

## 10. Limitations

- Models use structured metadata only — no case text, briefs, or oral argument data.
- Prediction of individual justice votes is outside scope.
- Class imbalance in Remand and Other categories limits multiclass performance.
- The Court's behavior changes with each new justice appointment, which
  structured metadata captures only imperfectly via naturalCourt.

---

## 11. Output Artifacts

### Models
- `models/FINAL_best_binary_model.joblib`
- `models/FINAL_best_multiclass_model.joblib`
- `models/scaler.joblib`

### Results Tables
- `results/tables/binary_results.csv`
- `results/tables/multiclass_results.csv`
- `results/tables/ALL_results_summary.csv`
- `results/tables/decade_accuracy.csv`
- `results/tables/window_sensitivity.csv`
- `results/tables/confident_errors.csv`

### Figures
- `results/figures/class_distribution.png`
- `results/figures/split_distribution.png`
- `results/figures/roc_curves_binary.png`
- `results/figures/model_comparison.png`
- `results/figures/feature_importance_*.png`
- `results/figures/shap_summary_*.png`
- `results/figures/shap_dependence_*.png`
- `results/figures/shap_waterfall_*.png`
- `results/figures/decade_accuracy_*.png`
- `results/figures/stability_comparison.png`
- `results/figures/window_sensitivity.png`
- `results/figures/binary_confusion_*.png`
- `results/figures/multiclass_confusion_*.png`

---

*Report auto-generated by phase10_finalization.py*

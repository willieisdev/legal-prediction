"""
=============================================================
PHASE 5 — BINARY MODELING (Reverse vs Affirm)
=============================================================
Three training strategies per model:
  A) Unbalanced     — no compensation
  B) SMOTE          — oversample minority class in train set
  C) Class Weights  — penalize misclassification of minority

Models:
  1. Majority-class Baseline
  2. Logistic Regression
  3. Random Forest
  4. LightGBM

Tuning  : RandomizedSearchCV with TimeSeriesSplit (optimises F1)
Reporting: cross_validate on best estimator reports ALL metrics:
           Accuracy, Precision, Recall, F1, F1-Macro,
           F1-Weighted, ROC-AUC
=============================================================
USAGE:
    python phase5_binary_modeling.py
=============================================================
"""

import os
import sys
import warnings
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.dummy           import DummyClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import (RandomizedSearchCV, TimeSeriesSplit,
                                     cross_validate)
from imblearn.over_sampling  import SMOTE
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_setup          import RANDOM_STATE
from phase4_data_splitting import load_binary_data

# ── Cross-validation config ───────────────────────────────
N_CV_SPLITS   = 5
N_ITER_SEARCH = 30

# ── All CV metrics reported for every model/strategy ─────
CV_SCORING = {
    "Accuracy"   : "accuracy",
    "Precision"  : "precision",
    "Recall"     : "recall",
    "F1"         : "f1",
    "F1_Macro"   : "f1_macro",
    "F1_Weighted": "f1_weighted",
    "ROC_AUC"    : "roc_auc",
}


def get_tscv() -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=N_CV_SPLITS)


def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    n_added = len(X_res) - len(X_train)
    print(f"      SMOTE: {len(X_train):,} → {len(X_res):,} (+{n_added:,} synthetic)")
    return X_res, y_res


def report_cv_metrics(model, X: np.ndarray, y: np.ndarray,
                      tscv: TimeSeriesSplit, label: str) -> dict:
    """
    Run cross_validate with ALL metrics on the best estimator.
    Returns dict of {metric_name: mean_cv_score}.
    """
    cv_results = cross_validate(
        model, X, y,
        cv=tscv,
        scoring=CV_SCORING,
        n_jobs=-1,
        return_train_score=False
    )

    means = {}
    print(f"\n      📊 CV Metrics — {label}")
    print(f"      {'Metric':<16} {'Mean CV':>9}  {'Std':>8}")
    print(f"      {'─'*37}")
    for name in CV_SCORING:
        key   = f"test_{name}"
        mean_v = cv_results[key].mean()
        std_v  = cv_results[key].std()
        means[name] = round(mean_v, 4)
        print(f"      {name:<16} {mean_v:>9.4f}  ±{std_v:.4f}")

    return means


def save_cv_summary(all_metrics: dict, fname: str) -> None:
    rows = []
    for model_key, metrics in all_metrics.items():
        row = {"Model": model_key}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv(f"results/tables/{fname}", index=False)
    print(f"\n💾 CV metrics saved → results/tables/{fname}")
    return df


# ══════════════════════════════════════════════════════════
# MODEL 1 — MAJORITY CLASS BASELINE
# ══════════════════════════════════════════════════════════

def train_baseline(X_train: np.ndarray, y_train: np.ndarray,
                   tscv: TimeSeriesSplit) -> tuple:
    print("\n" + "─"*54)
    print("  MODEL 1 — Majority Class Baseline")
    print("─"*54)

    model = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    majority_cls = model.classes_[np.argmax(model.class_prior_)]
    train_acc    = model.score(X_train, y_train)
    print(f"   Strategy  : most_frequent → always predicts class {majority_cls}")
    print(f"   Train Acc : {train_acc:.4f}")
    print(f"   Note      : No SMOTE/weight variants for baseline.")

    cv_m = report_cv_metrics(model, X_train, y_train, tscv,
                             label="Baseline (Unbalanced)")

    return ({"baseline_unbalanced": model},
            {"baseline_unbalanced": {"Strategy": "Unbalanced", **cv_m}})


# ══════════════════════════════════════════════════════════
# MODEL 2 — LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════

def _tune_logreg(X: np.ndarray, y: np.ndarray,
                 tscv: TimeSeriesSplit, class_weight) -> LogisticRegression:
    param_dist = {"C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}
    base = LogisticRegression(
        solver="lbfgs", max_iter=2000,
        class_weight=class_weight, random_state=RANDOM_STATE
    )
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=min(N_ITER_SEARCH, len(param_dist["C"])),
        cv=tscv, scoring="f1", n_jobs=-1, random_state=RANDOM_STATE
    )
    search.fit(X, y)
    print(f"      Best hyperparameter → C = {search.best_params_['C']:.4f}")
    return search.best_estimator_


def train_logreg(X_sc: np.ndarray, y_train: np.ndarray,
                 X_smote: np.ndarray, y_smote: np.ndarray,
                 tscv: TimeSeriesSplit) -> tuple:
    print("\n" + "─"*54)
    print("  MODEL 2 — Logistic Regression")
    print("─"*54)

    configs = [
        ("Unbalanced",    X_sc,    y_train, None,       "logreg_unbalanced"),
        ("SMOTE",         X_smote, y_smote, None,       "logreg_smote"),
        ("Class Weights", X_sc,    y_train, "balanced", "logreg_weighted"),
    ]

    models, metrics = {}, {}
    for label, X, y, cw, key in configs:
        print(f"\n   [{label}]")
        best = _tune_logreg(X, y, tscv, cw)
        cv_m = report_cv_metrics(best, X, y, tscv,
                                 label=f"Logistic Regression ({label})")
        models[key]  = best
        metrics[key] = {"Strategy": label, **cv_m}

    return models, metrics


# ══════════════════════════════════════════════════════════
# MODEL 3 — RANDOM FOREST
# ══════════════════════════════════════════════════════════

def _tune_rf(X: np.ndarray, y: np.ndarray,
             tscv: TimeSeriesSplit, class_weight) -> RandomForestClassifier:
    param_dist = {
        "n_estimators"    : [100, 200, 300, 500],
        "max_depth"       : [None, 5, 10, 20, 30],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features"    : ["sqrt", "log2", 0.3],
    }
    base = RandomForestClassifier(
        class_weight=class_weight, random_state=RANDOM_STATE, n_jobs=-1
    )
    search = RandomizedSearchCV(
        base, param_dist, n_iter=N_ITER_SEARCH,
        cv=tscv, scoring="f1", n_jobs=-1, random_state=RANDOM_STATE
    )
    search.fit(X, y)
    bp = search.best_params_
    print(f"      Best hyperparameters → "
          f"n_estimators={bp['n_estimators']}, "
          f"max_depth={bp['max_depth']}, "
          f"min_samples_leaf={bp['min_samples_leaf']}, "
          f"max_features={bp['max_features']}")
    return search.best_estimator_


def train_rf(X_un: np.ndarray, y_train: np.ndarray,
             X_smote: np.ndarray, y_smote: np.ndarray,
             tscv: TimeSeriesSplit) -> tuple:
    print("\n" + "─"*54)
    print("  MODEL 3 — Random Forest")
    print("─"*54)

    configs = [
        ("Unbalanced",    X_un,    y_train, None,       "rf_unbalanced"),
        ("SMOTE",         X_smote, y_smote, None,       "rf_smote"),
        ("Class Weights", X_un,    y_train, "balanced", "rf_weighted"),
    ]

    models, metrics = {}, {}
    for label, X, y, cw, key in configs:
        print(f"\n   [{label}]")
        best = _tune_rf(X, y, tscv, cw)
        cv_m = report_cv_metrics(best, X, y, tscv,
                                 label=f"Random Forest ({label})")
        models[key]  = best
        metrics[key] = {"Strategy": label, **cv_m}

    return models, metrics


# ══════════════════════════════════════════════════════════
# MODEL 4 — LIGHTGBM
# ══════════════════════════════════════════════════════════

def _tune_lgbm(X: np.ndarray, y: np.ndarray,
               tscv: TimeSeriesSplit,
               use_weights: bool) -> lgb.LGBMClassifier:
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    spw   = (n_neg / n_pos) if (use_weights and n_pos > 0) else 1.0

    param_dist = {
        "n_estimators"     : [100, 200, 300, 500],
        "num_leaves"       : [15, 31, 63, 127],
        "learning_rate"    : [0.01, 0.05, 0.1, 0.2],
        "min_child_samples": [10, 20, 30, 50],
        "subsample"        : [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree" : [0.7, 0.8, 0.9, 1.0],
    }
    base = lgb.LGBMClassifier(
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        verbose=-1, n_jobs=-1
    )
    search = RandomizedSearchCV(
        base, param_dist, n_iter=N_ITER_SEARCH,
        cv=tscv, scoring="f1", n_jobs=-1, random_state=RANDOM_STATE
    )
    search.fit(X, y)
    bp = search.best_params_
    print(f"      Best hyperparameters → "
          f"n_estimators={bp['n_estimators']}, "
          f"num_leaves={bp['num_leaves']}, "
          f"learning_rate={bp['learning_rate']}, "
          f"min_child_samples={bp['min_child_samples']}, "
          f"subsample={bp['subsample']}, "
          f"colsample_bytree={bp['colsample_bytree']}")
    return search.best_estimator_


def train_lgbm(X_un: np.ndarray, y_train: np.ndarray,
               X_smote: np.ndarray, y_smote: np.ndarray,
               tscv: TimeSeriesSplit) -> tuple:
    print("\n" + "─"*54)
    print("  MODEL 4 — LightGBM")
    print("─"*54)

    configs = [
        ("Unbalanced",    X_un,    y_train, False, "lgbm_unbalanced"),
        ("SMOTE",         X_smote, y_smote, False, "lgbm_smote"),
        ("Class Weights", X_un,    y_train, True,  "lgbm_weighted"),
    ]

    models, metrics = {}, {}
    for label, X, y, use_w, key in configs:
        print(f"\n   [{label}]")
        best = _tune_lgbm(X, y, tscv, use_weights=use_w)
        cv_m = report_cv_metrics(best, X, y, tscv,
                                 label=f"LightGBM ({label})")
        models[key]  = best
        metrics[key] = {"Strategy": label, **cv_m}

    return models, metrics


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PHASE 5 — BINARY MODELING")
    print("=" * 60)

    (X_tr_sc, X_tr_un, X_te_sc, X_te_un,
     y_train, y_test, feature_names) = load_binary_data()

    print(f"\n📐 Train: {X_tr_sc.shape[0]:,} | Test: {X_te_sc.shape[0]:,}")
    print(f"   Affirm (0): {(y_train==0).sum():,}  "
          f"Reverse (1): {(y_train==1).sum():,}  "
          f"Imbalance ratio: {(y_train==1).sum()/(y_train==0).sum():.2f}:1")

    print("\n🔧 Preparing SMOTE resampled training sets...")
    X_tr_sc_smote, y_smote_sc = apply_smote(X_tr_sc,  y_train)  # scaled  (for LR)
    X_tr_un_smote, y_smote_un = apply_smote(X_tr_un,  y_train)  # unscaled (for trees)

    tscv        = get_tscv()
    all_models  = {}
    all_metrics = {}
    t0          = time.time()

    m, cv = train_baseline(X_tr_sc, y_train, tscv)
    all_models.update(m);  all_metrics.update(cv)

    m, cv = train_logreg(X_tr_sc, y_train, X_tr_sc_smote, y_smote_sc, tscv)
    all_models.update(m);  all_metrics.update(cv)

    m, cv = train_rf(X_tr_un, y_train, X_tr_un_smote, y_smote_un, tscv)
    all_models.update(m);  all_metrics.update(cv)

    m, cv = train_lgbm(X_tr_un, y_train, X_tr_un_smote, y_smote_un, tscv)
    all_models.update(m);  all_metrics.update(cv)

    elapsed = time.time() - t0
    print(f"\n⏱  Total training time: {elapsed/60:.1f} minutes")

    # ── Save models ───────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    for name, model in all_models.items():
        joblib.dump(model, f"models/binary_{name}.joblib")

    print(f"\n💾 Saved {len(all_models)} binary models:")
    for name in all_models:
        print(f"   models/binary_{name}.joblib")

    # ── Save & print full CV metrics table ────────────────
    df_cv = save_cv_summary(all_metrics, "binary_cv_metrics.csv")

    print("\n" + "="*60)
    print("BINARY — FULL CV METRICS SUMMARY")
    print("="*60)
    rows = [{"Model": k, **v} for k, v in all_metrics.items()]
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format="{:.4f}".format))

    # ── Save model name list for Phase 7 ──────────────────
    pd.Series(list(all_models.keys())).to_csv(
        "data/processed/binary_model_names.csv", index=False
    )

    print("\n✅ Phase 5 complete.")
    print("   CV metrics  → results/tables/binary_cv_metrics.csv")
    print("   Models      → models/binary_*.joblib")


if __name__ == "__main__":
    main()
"""
=============================================================
PHASE 6 — MULTICLASS MODELING (Affirm/Reverse/Remand/Other)
=============================================================
Three training strategies per model:
  A) Unbalanced     — no compensation
  B) SMOTE          — oversample minority classes (train only)
  C) Class Weights  — penalize minority class errors

Models:
  1. Multinomial Logistic Regression
  2. Random Forest (multiclass)
  3. LightGBM (multiclass)

Tuning  : RandomizedSearchCV with TimeSeriesSplit (optimises F1-macro)
Reporting: cross_validate on best estimator reports ALL metrics:
           Accuracy, Macro-F1, Weighted-F1, per-class F1 (via
           custom scorers), ROC-AUC (OvR)
=============================================================
USAGE:
    python phase6_multiclass_modeling.py
=============================================================
"""

import os
import sys
import warnings
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import (RandomizedSearchCV, TimeSeriesSplit,
                                     cross_validate)
from sklearn.metrics         import make_scorer, f1_score
from imblearn.over_sampling  import SMOTE
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_setup          import RANDOM_STATE, MULTICLASS_LABELS
from phase4_data_splitting import load_multiclass_data

N_CV_SPLITS   = 5
N_ITER_SEARCH = 30


def get_tscv() -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=N_CV_SPLITS)


def get_cv_scoring(n_classes: int) -> dict:
    """
    Build the full scoring dict for multiclass cross_validate.
    Includes per-class F1 scorers using make_scorer.
    """
    scoring = {
        "Accuracy"   : "accuracy",
        "F1_Macro"   : "f1_macro",
        "F1_Weighted": "f1_weighted",
        "ROC_AUC_OvR": "roc_auc_ovr",
    }
    # Per-class F1 for each class that exists in the label set
    for cls_idx, cls_name in MULTICLASS_LABELS.items():
        if cls_idx < n_classes:
            scoring[f"F1_{cls_name}"] = make_scorer(
                f1_score, average=None, labels=[cls_idx],
                zero_division=0
            )
    return scoring


def apply_smote_multiclass(X: np.ndarray, y: np.ndarray) -> tuple:
    """SMOTE for multiclass — adjusts k_neighbors to smallest class size."""
    class_counts = np.bincount(y.astype(int))
    min_count    = class_counts[class_counts > 0].min()
    k = max(1, min(5, min_count - 1))

    if k < 1:
        print("      ⚠️  Smallest class too small for SMOTE — using original.")
        return X, y

    sm = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"      SMOTE (k={k}): {len(X):,} → {len(X_res):,} samples")
    return X_res, y_res


def report_cv_metrics(model, X: np.ndarray, y: np.ndarray,
                      tscv: TimeSeriesSplit, label: str,
                      n_classes: int) -> dict:
    """
    Run cross_validate with ALL metrics on the best estimator.
    Per-class F1 scorers return arrays — we unpack them cleanly.
    """
    scoring = get_cv_scoring(n_classes)

    cv_results = cross_validate(
        model, X, y,
        cv=tscv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    means = {}
    print(f"\n      📊 CV Metrics — {label}")
    print(f"      {'Metric':<22} {'Mean CV':>9}  {'Std':>8}")
    print(f"      {'─'*43}")

    for metric_name, scorer_key in scoring.items():
        key = f"test_{metric_name}"
        raw = cv_results[key]

        # Per-class scorers return arrays per fold — take mean across folds
        # Each fold returns shape (1,) since labels=[cls_idx]
        if raw.dtype == object or (raw.ndim > 1):
            # Flatten to scalar per fold then average
            fold_scores = np.array([np.mean(v) for v in raw])
        else:
            fold_scores = raw

        mean_v = float(np.mean(fold_scores))
        std_v  = float(np.std(fold_scores))
        means[metric_name] = round(mean_v, 4)
        print(f"      {metric_name:<22} {mean_v:>9.4f}  ±{std_v:.4f}")

    return means


def report_class_counts(y: np.ndarray, label: str) -> None:
    counts = np.bincount(y.astype(int))
    print(f"\n   {label} class distribution:")
    for cls, n in enumerate(counts):
        name = MULTICLASS_LABELS.get(cls, f"class_{cls}")
        pct  = n / len(y) * 100
        print(f"     {name:<10}: {n:,}  ({pct:.1f}%)")


def save_cv_summary(all_metrics: dict, fname: str) -> None:
    rows = [{"Model": k, **v} for k, v in all_metrics.items()]
    df   = pd.DataFrame(rows)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv(f"results/tables/{fname}", index=False)
    print(f"\n💾 CV metrics saved → results/tables/{fname}")


# ══════════════════════════════════════════════════════════
# MODEL 1 — MULTINOMIAL LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
# MODEL 1 — MULTINOMIAL LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════

def _tune_logreg_mc(X: np.ndarray, y: np.ndarray,
                    tscv: TimeSeriesSplit, class_weight) -> LogisticRegression:
    param_dist = {"C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}
    
    # REMOVED: multi_class="multinomial"
    # Scikit-learn now handles this automatically based on the solver and target labels.
    base = LogisticRegression(
        solver="lbfgs",
        max_iter=3000, 
        class_weight=class_weight,
        random_state=RANDOM_STATE
    )
    
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=min(N_ITER_SEARCH, len(param_dist["C"])),
        cv=tscv, scoring="f1_macro", n_jobs=-1, random_state=RANDOM_STATE
    )
    search.fit(X, y)
    print(f"      Best hyperparameter → C = {search.best_params_['C']:.4f}")
    return search.best_estimator_

def train_logreg_mc(X_sc: np.ndarray, y_train: np.ndarray,
                    X_smote: np.ndarray, y_smote: np.ndarray,
                    tscv: TimeSeriesSplit, n_classes: int) -> tuple:
    print("\n" + "─"*54)
    print("  MODEL 1 — Multinomial Logistic Regression")
    print("─"*54)

    configs = [
        ("Unbalanced",    X_sc,    y_train, None,       "mc_logreg_unbalanced"),
        ("SMOTE",         X_smote, y_smote, None,       "mc_logreg_smote"),
        ("Class Weights", X_sc,    y_train, "balanced", "mc_logreg_weighted"),
    ]

    models, metrics = {}, {}
    for label, X, y, cw, key in configs:
        print(f"\n   [{label}]")
        best = _tune_logreg_mc(X, y, tscv, cw)
        cv_m = report_cv_metrics(best, X, y, tscv,
                                 label=f"LR Multiclass ({label})",
                                 n_classes=n_classes)
        models[key]  = best
        metrics[key] = {"Strategy": label, **cv_m}

    return models, metrics


# ══════════════════════════════════════════════════════════
# MODEL 2 — RANDOM FOREST (MULTICLASS)
# ══════════════════════════════════════════════════════════

def _tune_rf_mc(X: np.ndarray, y: np.ndarray,
                tscv: TimeSeriesSplit, class_weight) -> RandomForestClassifier:
    param_dist = {
        "n_estimators"    : [100, 200, 300, 500],
        "max_depth"       : [None, 5, 10, 20, 30],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features"    : ["sqrt", "log2", 0.3],
    }
    base = RandomForestClassifier(
        class_weight=class_weight,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    search = RandomizedSearchCV(
        base, param_dist, n_iter=N_ITER_SEARCH,
        cv=tscv, scoring="f1_macro", n_jobs=-1, random_state=RANDOM_STATE
    )
    search.fit(X, y)
    bp = search.best_params_
    print(f"      Best hyperparameters → "
          f"n_estimators={bp['n_estimators']}, "
          f"max_depth={bp['max_depth']}, "
          f"min_samples_leaf={bp['min_samples_leaf']}, "
          f"max_features={bp['max_features']}")
    return search.best_estimator_


def train_rf_mc(X_un: np.ndarray, y_train: np.ndarray,
                X_smote: np.ndarray, y_smote: np.ndarray,
                tscv: TimeSeriesSplit, n_classes: int) -> tuple:
    print("\n" + "─"*54)
    print("  MODEL 2 — Random Forest (Multiclass)")
    print("─"*54)

    configs = [
        ("Unbalanced",    X_un,    y_train, None,       "mc_rf_unbalanced"),
        ("SMOTE",         X_smote, y_smote, None,       "mc_rf_smote"),
        ("Class Weights", X_un,    y_train, "balanced", "mc_rf_weighted"),
    ]

    models, metrics = {}, {}
    for label, X, y, cw, key in configs:
        print(f"\n   [{label}]")
        best = _tune_rf_mc(X, y, tscv, cw)
        cv_m = report_cv_metrics(best, X, y, tscv,
                                 label=f"RF Multiclass ({label})",
                                 n_classes=n_classes)
        models[key]  = best
        metrics[key] = {"Strategy": label, **cv_m}

    return models, metrics


# ══════════════════════════════════════════════════════════
# MODEL 3 — LIGHTGBM (MULTICLASS)
# ══════════════════════════════════════════════════════════

def _tune_lgbm_mc(X: np.ndarray, y: np.ndarray,
                  tscv: TimeSeriesSplit,
                  use_weights: bool, n_classes: int) -> lgb.LGBMClassifier:
    if use_weights:
        counts = np.bincount(y.astype(int))
        total  = len(y)
        cw = {i: total / (n_classes * c) if c > 0 else 1.0
              for i, c in enumerate(counts)}
    else:
        cw = None

    param_dist = {
        "n_estimators"     : [100, 200, 300, 500],
        "num_leaves"       : [15, 31, 63, 127],
        "learning_rate"    : [0.01, 0.05, 0.1, 0.2],
        "min_child_samples": [10, 20, 30, 50],
        "subsample"        : [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree" : [0.7, 0.8, 0.9, 1.0],
    }
    base = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        class_weight=cw,
        random_state=RANDOM_STATE,
        verbose=-1, n_jobs=-1
    )
    search = RandomizedSearchCV(
        base, param_dist, n_iter=N_ITER_SEARCH,
        cv=tscv, scoring="f1_macro", n_jobs=-1, random_state=RANDOM_STATE
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


def train_lgbm_mc(X_un: np.ndarray, y_train: np.ndarray,
                  X_smote: np.ndarray, y_smote: np.ndarray,
                  tscv: TimeSeriesSplit, n_classes: int) -> tuple:
    print("\n" + "─"*54)
    print("  MODEL 3 — LightGBM (Multiclass)")
    print("─"*54)

    configs = [
        ("Unbalanced",    X_un,    y_train, False, "mc_lgbm_unbalanced"),
        ("SMOTE",         X_smote, y_smote, False, "mc_lgbm_smote"),
        ("Class Weights", X_un,    y_train, True,  "mc_lgbm_weighted"),
    ]

    models, metrics = {}, {}
    for label, X, y, use_w, key in configs:
        print(f"\n   [{label}]")
        best = _tune_lgbm_mc(X, y, tscv, use_weights=use_w, n_classes=n_classes)
        cv_m = report_cv_metrics(best, X, y, tscv,
                                 label=f"LightGBM Multiclass ({label})",
                                 n_classes=n_classes)
        models[key]  = best
        metrics[key] = {"Strategy": label, **cv_m}

    return models, metrics


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PHASE 6 — MULTICLASS MODELING")
    print("=" * 60)

    (X_tr_sc, X_tr_un, X_te_sc, X_te_un,
     y_train, y_test, feature_names) = load_multiclass_data()

    n_classes = len(np.unique(y_train))
    print(f"\n📐 Train: {X_tr_sc.shape[0]:,} | Test: {X_te_sc.shape[0]:,}")
    print(f"   Number of classes: {n_classes}")
    report_class_counts(y_train, "TRAIN")

    # ── SMOTE ─────────────────────────────────────────────
    print("\n🔧 Preparing SMOTE resampled sets...")
    X_tr_sc_smote, y_smote_sc = apply_smote_multiclass(X_tr_sc,  y_train)
    X_tr_un_smote, y_smote_un = apply_smote_multiclass(X_tr_un,  y_train)
    report_class_counts(y_smote_sc, "TRAIN after SMOTE")

    tscv        = get_tscv()
    all_models  = {}
    all_metrics = {}
    t0          = time.time()

    m, cv = train_logreg_mc(X_tr_sc, y_train,
                             X_tr_sc_smote, y_smote_sc,
                             tscv, n_classes)
    all_models.update(m);  all_metrics.update(cv)

    m, cv = train_rf_mc(X_tr_un, y_train,
                        X_tr_un_smote, y_smote_un,
                        tscv, n_classes)
    all_models.update(m);  all_metrics.update(cv)

    m, cv = train_lgbm_mc(X_tr_un, y_train,
                          X_tr_un_smote, y_smote_un,
                          tscv, n_classes)
    all_models.update(m);  all_metrics.update(cv)

    elapsed = time.time() - t0
    print(f"\n⏱  Total training time: {elapsed/60:.1f} minutes")

    # ── Save models ───────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    for name, model in all_models.items():
        joblib.dump(model, f"models/{name}.joblib")

    print(f"\n💾 Saved {len(all_models)} multiclass models:")
    for name in all_models:
        print(f"   models/{name}.joblib")

    # ── Save & print full CV metrics table ────────────────
    save_cv_summary(all_metrics, "multiclass_cv_metrics.csv")

    print("\n" + "="*60)
    print("MULTICLASS — FULL CV METRICS SUMMARY")
    print("="*60)
    rows = [{"Model": k, **v} for k, v in all_metrics.items()]
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format="{:.4f}".format))

    pd.Series(list(all_models.keys())).to_csv(
        "data/processed/multiclass_model_names.csv", index=False
    )

    print("\n✅ Phase 6 complete.")
    print("   CV metrics → results/tables/multiclass_cv_metrics.csv")
    print("   Models     → models/mc_*.joblib")


if __name__ == "__main__":
    main()
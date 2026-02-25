"""
=============================================================
PHASE 3 — FEATURE ENGINEERING
=============================================================
- Select pre-decision features only
- Remove leakage variables
- Frequency-encode high-cardinality categoricals
- One-hot encode medium-cardinality categoricals
- Handle missing values (fit on train, transform test)
- Scale numerical features
- Save processed feature matrices
=============================================================
USAGE:
    python phase3_feature_engineering.py
    (Reads data/processed/scdb_prepared.csv)
=============================================================
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_setup import (
    PRE_DECISION_FEATURES, LEAKAGE_VARS, DROP_COLS,
    TRAIN_END_YEAR, TEST_START_YEAR, RANDOM_STATE
)

# ── Feature type definitions ──────────────────────────────
# One-hot: medium cardinality (≤ 20 unique values)
ONE_HOT_FEATURES = [
    "issueArea",            # 14 categories
    "jurisdiction",         # ~15 categories
    "certReason",           # ~13 categories
    "lcDispositionDirection",  # 3 categories: 1=liberal, 2=conservative, 3=mixed
    "decisionType",         # ~7 categories
    "threeJudgeFdc",        # binary 0/1
    "lcDisagreement",       # binary 0/1
    "authorityDecision1",   # ~7 categories
    "lawType",              # ~9 categories
]

# Frequency-encode: high cardinality (many unique codes)
FREQ_ENCODE_FEATURES = [
    "petitioner",           # ~300 codes
    "respondent",           # ~300 codes
    "naturalCourt",         # ~40 court compositions
    "caseOrigin",           # ~300 court codes
    "caseSource",           # ~300 court codes
    "petitionerState",      # ~61 states (sparse)
    "respondentState",      # ~61 states (sparse)
    "caseOriginState",      # ~61 states (sparse)
    "caseSourceState",      # ~61 states (sparse)
    "adminAction",          # ~124 agency codes (sparse)
    "lcDisposition",        # ~12 codes
]

# Numerical: keep as-is then scale
NUMERICAL_FEATURES = [
    "term",                 # year 1946–2024
]


def load_prepared_data() -> pd.DataFrame:
    path = "data/processed/scdb_prepared.csv"
    print(f"\n📂 Loading prepared data from: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only pre-decision features + targets + term for splitting."""
    # Build deduplicated list preserving order
    all_keep = PRE_DECISION_FEATURES + ["target_binary", "target_multi", "term"]
    seen = set()
    deduped = []
    for c in all_keep:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    # Only keep columns that actually exist in dataframe
    keep = [c for c in deduped if c in df.columns]
    dropped_requested = [c for c in deduped if c not in df.columns]
    if dropped_requested:
        print(f"\n⚠️  Requested features not found in data: {dropped_requested}")
    df = df[keep].copy()
    # Final safety check — drop any remaining duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"\n✅ Selected {len(df.columns)} columns (features + targets)")
    return df


def report_missingness(df: pd.DataFrame, features: list) -> None:
    """Print missing value rates for all feature columns."""
    print("\n📊 Missing value rates (feature columns only):")
    miss = df[features].isna().mean().sort_values(ascending=False)
    for col, rate in miss.items():
        if rate > 0:
            print(f"   {col:<30} : {rate*100:.1f}% missing")
    if (miss == 0).all():
        print("   No missing values in feature set.")


def frequency_encode(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     col: str) -> tuple:
    """
    Replace category codes with their frequency (proportion) in training data.
    Unknown codes in test set get frequency = 0.
    NaN is treated as a separate category 'MISSING' before encoding.
    """
    train_df = train_df.copy()
    test_df  = test_df.copy()

    # Fill NaN with sentinel -999 so it participates in freq map
    train_df[col] = train_df[col].fillna(-999)
    test_df[col]  = test_df[col].fillna(-999)

    freq_map = (train_df[col].value_counts(normalize=True)
                              .to_dict())

    train_df[col] = train_df[col].map(freq_map).fillna(0.0)
    test_df[col]  = test_df[col].map(freq_map).fillna(0.0)

    return train_df, test_df


def build_features(df: pd.DataFrame) -> tuple:
    """
    Full feature engineering pipeline.
    Returns:
        X_train_scaled, X_test_scaled  : scaled feature matrices (numpy)
        X_train_unscaled, X_test_unscaled : unscaled (for tree models)
        y_train_bin, y_test_bin        : binary targets
        y_train_multi, y_test_multi    : multiclass targets
        feature_names                  : list of final feature names
        scaler                         : fitted StandardScaler
    """
    print("\n" + "="*50)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*50)

    # ── Step 1: Select features (deduplicated) ────────────
    wanted = []
    seen   = set()
    for f in ONE_HOT_FEATURES + FREQ_ENCODE_FEATURES + NUMERICAL_FEATURES \
             + ["target_binary", "target_multi", "term"]:
        if f not in seen:
            seen.add(f)
            wanted.append(f)

    features_available = [f for f in wanted if f in df.columns]
    df_feat = df[features_available].copy()
    # Safety: drop duplicate columns if any slipped through
    df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()].reset_index(drop=True)

    # ── Step 2: Chronological split using explicit boolean mask ──
    term_col     = df_feat["term"].values
    train_mask   = term_col <= TRAIN_END_YEAR
    test_mask    = term_col >= TEST_START_YEAR
    train_df = df_feat.iloc[train_mask].copy().reset_index(drop=True)
    test_df  = df_feat.iloc[test_mask].copy().reset_index(drop=True)
    print(f"\n📅 Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")

    # ── Step 3: Extract targets ────────────────────────────
    y_train_bin   = train_df["target_binary"].copy()
    y_test_bin    = test_df["target_binary"].copy()
    y_train_multi = train_df["target_multi"].copy()
    y_test_multi  = test_df["target_multi"].copy()

    # Drop target columns from feature frames
    drop_from_feat = ["target_binary", "target_multi"]
    train_df = train_df.drop(columns=drop_from_feat)
    test_df  = test_df.drop(columns=drop_from_feat)

    # ── Step 4: Frequency encoding (fit on train only) ────
    freq_feats_present = [f for f in FREQ_ENCODE_FEATURES if f in train_df.columns]
    print(f"\n🔧 Frequency-encoding {len(freq_feats_present)} high-cardinality features...")
    for col in freq_feats_present:
        train_df, test_df = frequency_encode(train_df, test_df, col)

    # ── Step 5: Fill NaN in one-hot features before encoding
    oh_feats_present = [f for f in ONE_HOT_FEATURES if f in train_df.columns]
    for col in oh_feats_present:
        # Fill with -1 sentinel so encoder creates an "Unknown" column
        train_df[col] = train_df[col].fillna(-1).astype(int).astype(str)
        test_df[col]  = test_df[col].fillna(-1).astype(int).astype(str)

    # ── Step 6: One-hot encoding ───────────────────────────
    print(f"🔧 One-hot encoding {len(oh_feats_present)} medium-cardinality features...")

    # Fit on train, align test to same columns
    train_dummies = pd.get_dummies(train_df[oh_feats_present], prefix=oh_feats_present)
    test_dummies  = pd.get_dummies(test_df[oh_feats_present],  prefix=oh_feats_present)

    # Align test dummies to train dummies (add missing cols as 0, drop extra)
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

    # Drop original one-hot columns and concatenate encoded
    train_df = train_df.drop(columns=oh_feats_present)
    test_df  = test_df.drop(columns=oh_feats_present)
    train_df = pd.concat([train_df.reset_index(drop=True),
                          train_dummies.reset_index(drop=True)], axis=1)
    test_df  = pd.concat([test_df.reset_index(drop=True),
                          test_dummies.reset_index(drop=True)], axis=1)

    # ── Step 7: Fill remaining NaN in numerical features ──
    num_feats_present = [f for f in NUMERICAL_FEATURES if f in train_df.columns]
    for col in num_feats_present:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col]  = test_df[col].fillna(median_val)

    # ── Step 8: Drop term (used for split only) ───────────
    if "term" in train_df.columns:
        train_df = train_df.drop(columns=["term"])
        test_df  = test_df.drop(columns=["term"])

    # ── Step 9: Ensure all columns are numeric ─────────────
    train_df = train_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    test_df  = test_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    feature_names = list(train_df.columns)
    print(f"\n✅ Final feature matrix: {len(feature_names)} features")

    # ── Step 10: Scale (fit on train only) ────────────────
    print("🔧 Scaling features (StandardScaler fit on train only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df.values)
    X_test_scaled  = scaler.transform(test_df.values)

    # Unscaled versions for tree-based models
    X_train_unscaled = train_df.values.astype(np.float32)
    X_test_unscaled  = test_df.values.astype(np.float32)

    return (X_train_scaled, X_test_scaled,
            X_train_unscaled, X_test_unscaled,
            y_train_bin, y_test_bin,
            y_train_multi, y_test_multi,
            feature_names, scaler)


def report_feature_summary(feature_names: list, X_train: np.ndarray) -> None:
    print(f"\n📊 Feature matrix summary:")
    print(f"   Total features : {len(feature_names)}")
    print(f"   Train samples  : {X_train.shape[0]:,}")
    print(f"   Non-zero rate  : {(X_train != 0).mean() * 100:.1f}%")


def save_artifacts(X_train_scaled, X_test_scaled,
                   X_train_unscaled, X_test_unscaled,
                   y_train_bin, y_test_bin,
                   y_train_multi, y_test_multi,
                   feature_names, scaler) -> None:
    """Save all processed arrays and the scaler."""
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/X_train_scaled.npy",   X_train_scaled)
    np.save("data/processed/X_test_scaled.npy",    X_test_scaled)
    np.save("data/processed/X_train_unscaled.npy", X_train_unscaled)
    np.save("data/processed/X_test_unscaled.npy",  X_test_unscaled)

    # Save targets — drop NaN rows per task
    pd.Series(y_train_bin.values,   name="target_binary").to_csv("data/processed/y_train_bin.csv",   index=False)
    pd.Series(y_test_bin.values,    name="target_binary").to_csv("data/processed/y_test_bin.csv",    index=False)
    pd.Series(y_train_multi.values, name="target_multi").to_csv("data/processed/y_train_multi.csv",  index=False)
    pd.Series(y_test_multi.values,  name="target_multi").to_csv("data/processed/y_test_multi.csv",   index=False)

    # Save feature names
    pd.Series(feature_names).to_csv("data/processed/feature_names.csv", index=False)

    # Save scaler
    joblib.dump(scaler, "models/scaler.joblib")

    print("\n💾 Saved artifacts:")
    print("   data/processed/X_train_scaled.npy")
    print("   data/processed/X_test_scaled.npy")
    print("   data/processed/X_train_unscaled.npy")
    print("   data/processed/X_test_unscaled.npy")
    print("   data/processed/y_train_bin.csv  |  y_test_bin.csv")
    print("   data/processed/y_train_multi.csv | y_test_multi.csv")
    print("   data/processed/feature_names.csv")
    print("   models/scaler.joblib")


def plot_feature_importance_preview(X_train: np.ndarray,
                                    feature_names: list,
                                    y_train: pd.Series) -> None:
    """Quick variance-based feature preview plot."""
    # Use variance as a proxy for potential importance
    variances = np.var(X_train, axis=0)
    top_n = 20
    top_idx = np.argsort(variances)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]
    top_vars  = variances[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), top_vars[::-1], color="#1976D2", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Variance (scaled features)")
    ax.set_title("Top 20 Features by Variance (Preview)", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/feature_variance_preview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Feature variance plot saved → results/figures/feature_variance_preview.png")


def main():
    print("=" * 60)
    print("PHASE 3 — FEATURE ENGINEERING")
    print("=" * 60)

    df = load_prepared_data()

    # Report missingness before engineering
    feat_cols_present = [f for f in
                         ONE_HOT_FEATURES + FREQ_ENCODE_FEATURES + NUMERICAL_FEATURES
                         if f in df.columns]
    report_missingness(df, feat_cols_present)

    (X_train_scaled, X_test_scaled,
     X_train_unscaled, X_test_unscaled,
     y_train_bin, y_test_bin,
     y_train_multi, y_test_multi,
     feature_names, scaler) = build_features(df)

    report_feature_summary(feature_names, X_train_scaled)
    plot_feature_importance_preview(X_train_scaled, feature_names, y_train_bin)

    save_artifacts(
        X_train_scaled, X_test_scaled,
        X_train_unscaled, X_test_unscaled,
        y_train_bin, y_test_bin,
        y_train_multi, y_test_multi,
        feature_names, scaler
    )

    print("\n✅ Phase 3 complete.")


if __name__ == "__main__":
    main()
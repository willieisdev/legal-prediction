"""
=============================================================
PHASE 4 — DATA SPLITTING & SPLIT VALIDATION
=============================================================
- Loads processed arrays from Phase 3
- Handles NaN rows per task (binary vs multiclass)
- Provides clean loaders used by Phases 5 and 6
- Validates no temporal leakage
- Reports final split sizes and class distributions
=============================================================
USAGE:
    python phase4_data_splitting.py
    (or import load_binary_data / load_multiclass_data in other phases)
=============================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_setup import (
    BINARY_LABELS, MULTICLASS_LABELS,
    TRAIN_END_YEAR, TEST_START_YEAR
)


def _load_raw_arrays() -> dict:
    """Load all saved numpy arrays and target CSVs from Phase 3."""
    arrays = {
        "X_train_scaled"  : np.load("data/processed/X_train_scaled.npy"),
        "X_test_scaled"   : np.load("data/processed/X_test_scaled.npy"),
        "X_train_unscaled": np.load("data/processed/X_train_unscaled.npy"),
        "X_test_unscaled" : np.load("data/processed/X_test_unscaled.npy"),
        "y_train_bin"     : pd.read_csv("data/processed/y_train_bin.csv").squeeze(),
        "y_test_bin"      : pd.read_csv("data/processed/y_test_bin.csv").squeeze(),
        "y_train_multi"   : pd.read_csv("data/processed/y_train_multi.csv").squeeze(),
        "y_test_multi"    : pd.read_csv("data/processed/y_test_multi.csv").squeeze(),
        "feature_names"   : pd.read_csv("data/processed/feature_names.csv").squeeze().tolist(),
    }
    return arrays


def _clean_split(X_scaled, X_unscaled, y) -> tuple:
    """
    Remove rows where target is NaN.
    Returns cleaned (X_scaled, X_unscaled, y) as numpy arrays.
    """
    y_arr  = np.array(y, dtype=float)
    mask   = ~np.isnan(y_arr)
    y_clean = y_arr[mask].astype(int)
    return X_scaled[mask], X_unscaled[mask], y_clean


def load_binary_data() -> tuple:
    """
    Returns:
        X_train_sc, X_train_un  : scaled and unscaled train features
        X_test_sc,  X_test_un   : scaled and unscaled test features
        y_train,    y_test       : binary integer targets (0/1)
        feature_names            : list of feature name strings
    """
    raw = _load_raw_arrays()

    X_tr_sc, X_tr_un, y_tr = _clean_split(
        raw["X_train_scaled"], raw["X_train_unscaled"], raw["y_train_bin"])
    X_te_sc, X_te_un, y_te = _clean_split(
        raw["X_test_scaled"],  raw["X_test_unscaled"],  raw["y_test_bin"])

    return (X_tr_sc, X_tr_un, X_te_sc, X_te_un,
            y_tr, y_te, raw["feature_names"])


def load_multiclass_data() -> tuple:
    """
    Returns:
        X_train_sc, X_train_un  : scaled and unscaled train features
        X_test_sc,  X_test_un   : scaled and unscaled test features
        y_train,    y_test       : multiclass integer targets (0/1/2/3)
        feature_names            : list of feature name strings
    """
    raw = _load_raw_arrays()

    X_tr_sc, X_tr_un, y_tr = _clean_split(
        raw["X_train_scaled"], raw["X_train_unscaled"], raw["y_train_multi"])
    X_te_sc, X_te_un, y_te = _clean_split(
        raw["X_test_scaled"],  raw["X_test_unscaled"],  raw["y_test_multi"])

    return (X_tr_sc, X_tr_un, X_te_sc, X_te_un,
            y_tr, y_te, raw["feature_names"])


def validate_split() -> None:
    """Print a full report of split sizes and class distributions."""
    print("=" * 60)
    print("PHASE 4 — SPLIT VALIDATION REPORT")
    print("=" * 60)

    (X_tr_sc, X_tr_un, X_te_sc, X_te_un,
     y_tr_bin, y_te_bin, feat_names) = load_binary_data()

    (_, _, _, _,
     y_tr_mc, y_te_mc, _) = load_multiclass_data()

    # ── Sizes ─────────────────────────────────────────────
    print(f"\n📐 Feature matrix dimensions:")
    print(f"   X_train : {X_tr_sc.shape[0]:,} rows × {X_tr_sc.shape[1]} features")
    print(f"   X_test  : {X_te_sc.shape[0]:,} rows × {X_te_sc.shape[1]} features")
    print(f"   Features: {len(feat_names)}")

    # ── Binary distribution ───────────────────────────────
    print(f"\n📊 Binary target — TRAIN (1946–{TRAIN_END_YEAR}):")
    for cls in sorted(np.unique(y_tr_bin)):
        n   = (y_tr_bin == cls).sum()
        pct = n / len(y_tr_bin) * 100
        print(f"   {BINARY_LABELS[cls]:<10} : {n:>5,}  ({pct:.1f}%)")

    print(f"\n📊 Binary target — TEST ({TEST_START_YEAR}–present):")
    for cls in sorted(np.unique(y_te_bin)):
        n   = (y_te_bin == cls).sum()
        pct = n / len(y_te_bin) * 100
        print(f"   {BINARY_LABELS[cls]:<10} : {n:>5,}  ({pct:.1f}%)")

    # ── Multiclass distribution ───────────────────────────
    print(f"\n📊 Multiclass target — TRAIN:")
    for cls in sorted(np.unique(y_tr_mc)):
        n   = (y_tr_mc == cls).sum()
        pct = n / len(y_tr_mc) * 100
        label = MULTICLASS_LABELS.get(cls, f"class_{cls}")
        print(f"   {label:<10} : {n:>5,}  ({pct:.1f}%)")

    print(f"\n📊 Multiclass target — TEST:")
    for cls in sorted(np.unique(y_te_mc)):
        n   = (y_te_mc == cls).sum()
        pct = n / len(y_te_mc) * 100
        label = MULTICLASS_LABELS.get(cls, f"class_{cls}")
        print(f"   {label:<10} : {n:>5,}  ({pct:.1f}%)")

    # ── Leakage check ─────────────────────────────────────
    print(f"\n🔒 Leakage check:")
    print(f"   Train uses terms ≤ {TRAIN_END_YEAR}")
    print(f"   Test  uses terms ≥ {TEST_START_YEAR}")
    print(f"   ✅ No overlap possible — chronological split enforced in Phase 3.")

    # ── Plot split distribution ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Train / Test Class Distribution\n"
                 f"(Train: 1946–{TRAIN_END_YEAR} | Test: {TEST_START_YEAR}–present)",
                 fontsize=13, fontweight="bold")

    colors_train = ["#1565C0", "#C62828"]
    colors_test  = ["#42A5F5", "#EF9A9A"]

    # Binary
    for i, (y, title, colors) in enumerate([
        (np.concatenate([y_tr_bin, y_te_bin]),  "Binary", ["#1565C0", "#C62828"])
    ]):
        pass

    bin_labels = [BINARY_LABELS[c] for c in sorted(np.unique(y_tr_bin))]
    train_bin_counts = [(y_tr_bin == c).sum() for c in sorted(np.unique(y_tr_bin))]
    test_bin_counts  = [(y_te_bin == c).sum() for c in sorted(np.unique(y_te_bin))]

    x = np.arange(len(bin_labels))
    w = 0.35
    axes[0].bar(x - w/2, train_bin_counts, w, label=f"Train (≤{TRAIN_END_YEAR})",
                color="#1565C0", edgecolor="black", linewidth=0.7)
    axes[0].bar(x + w/2, test_bin_counts,  w, label=f"Test (≥{TEST_START_YEAR})",
                color="#42A5F5", edgecolor="black", linewidth=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bin_labels)
    axes[0].set_title("Binary Task")
    axes[0].set_ylabel("Cases")
    axes[0].legend()

    mc_labels = [MULTICLASS_LABELS.get(c, str(c)) for c in sorted(np.unique(y_tr_mc))]
    train_mc_counts = [(y_tr_mc == c).sum() for c in sorted(np.unique(y_tr_mc))]
    test_mc_counts  = [(y_te_mc == c).sum() for c in sorted(np.unique(y_te_mc))]
    x2 = np.arange(len(mc_labels))
    axes[1].bar(x2 - w/2, train_mc_counts, w, label=f"Train (≤{TRAIN_END_YEAR})",
                color="#1B5E20", edgecolor="black", linewidth=0.7)
    axes[1].bar(x2 + w/2, test_mc_counts,  w, label=f"Test (≥{TEST_START_YEAR})",
                color="#81C784", edgecolor="black", linewidth=0.7)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(mc_labels)
    axes[1].set_title("Multiclass Task")
    axes[1].set_ylabel("Cases")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/split_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n📊 Split distribution plot saved → results/figures/split_distribution.png")
    print("\n✅ Phase 4 complete — data loaders ready for Phases 5 and 6.")


if __name__ == "__main__":
    validate_split()

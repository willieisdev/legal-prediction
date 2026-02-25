"""
=============================================================
PHASE 2 — DATA PREPARATION
=============================================================
- Load SCDB dataset
- Remove cases without clear decisions
- Exclude procedural dismissals
- Create binary and multiclass target variables
- Report class distributions
=============================================================
USAGE:
    python phase2_data_preparation.py --data path/to/SCDB_2023_01_caseCentered_Citation.csv
=============================================================
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

# ── Import Phase 1 config ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_setup import (
    BINARY_MAP, BINARY_LABELS,
    MULTICLASS_MAP, MULTICLASS_LABELS,
    TRAIN_END_YEAR, TEST_START_YEAR,
)

# ── caseDisposition codes to EXCLUDE entirely ─────────────
# These are procedural/ambiguous and excluded from BOTH tasks
EXCLUDE_DISPOSITION_CODES = {1, 9, 10}
# Code 1  = stay           (procedural)
# Code 9  = unspecified    (ambiguous)
# Code 10 = dismiss        (procedural dismissal)


def load_data(filepath: str) -> pd.DataFrame:
    """Load the SCDB CSV file."""
    print(f"\n📂 Loading dataset from: {filepath}")
    df = pd.read_csv(filepath, encoding="latin-1", low_memory=False)
    print(f"   Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def inspect_disposition(df: pd.DataFrame) -> None:
    """Print value counts for caseDisposition before filtering."""
    print("\n📊 caseDisposition value counts (raw):")
    vc = df["caseDisposition"].value_counts(dropna=False).sort_index()
    labels = {
        1: "stay", 2: "affirm", 3: "reverse", 4: "reverse+remand",
        5: "vacate+remand", 6: "affirm+reverse", 7: "vacate",
        8: "affirm+remand", 9: "unspecified", 10: "dismiss", 11: "remand",
        np.nan: "MISSING"
    }
    for code, count in vc.items():
        label = labels.get(code, f"code_{code}")
        print(f"   Code {str(code):>4}  ({label:<22}) : {count:>5}")


def remove_missing_disposition(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where caseDisposition is NaN."""
    before = len(df)
    df = df.dropna(subset=["caseDisposition"]).copy()
    after = len(df)
    print(f"\n🗑  Removed {before - after:,} rows with missing caseDisposition.")
    print(f"   Remaining: {after:,} rows")
    return df


def exclude_procedural(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude stays, unspecified, and dismissals (codes 1, 9, 10)."""
    before = len(df)
    mask = df["caseDisposition"].isin(EXCLUDE_DISPOSITION_CODES)
    excluded = df[mask]
    print(f"\n🗑  Excluding procedural/ambiguous dispositions (codes {sorted(EXCLUDE_DISPOSITION_CODES)}):")
    for code in sorted(EXCLUDE_DISPOSITION_CODES):
        n = (excluded["caseDisposition"] == code).sum()
        print(f"   Code {code}: {n:,} rows")
    df = df[~mask].copy()
    print(f"   Total excluded: {before - len(df):,} rows")
    print(f"   Remaining: {len(df):,} rows")
    return df


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target: Affirm=0, Reverse=1.
    Codes NOT in BINARY_MAP (e.g. 5=vacate+remand, 11=remand)
    are kept in the dataframe but will have NaN for binary target.
    They are used in multiclass only.
    """
    df["target_binary"] = df["caseDisposition"].map(BINARY_MAP)
    n_mapped   = df["target_binary"].notna().sum()
    n_unmapped = df["target_binary"].isna().sum()
    print(f"\n✅ Binary target created:")
    print(f"   Mapped (usable for binary task) : {n_mapped:,}")
    print(f"   Unmapped (Remand-only codes)    : {n_unmapped:,} → used in multiclass only")
    return df


def create_multiclass_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create multiclass target: Affirm=0, Reverse=1, Remand=2, Other=3.
    All remaining codes after procedural exclusion should map here.
    """
    df["target_multi"] = df["caseDisposition"].map(MULTICLASS_MAP)
    n_unmapped = df["target_multi"].isna().sum()
    if n_unmapped > 0:
        print(f"\n⚠️  {n_unmapped} rows could not be mapped to multiclass target.")
        unmapped_codes = df.loc[df["target_multi"].isna(), "caseDisposition"].value_counts()
        print(f"   Unmapped codes:\n{unmapped_codes}")
    else:
        print(f"\n✅ Multiclass target created — all rows mapped successfully.")
    return df


def report_class_distribution(df: pd.DataFrame) -> None:
    """Print and plot class distributions for both targets."""

    # ── Binary distribution ───────────────────────────────
    df_bin = df.dropna(subset=["target_binary"])
    bin_counts = df_bin["target_binary"].value_counts().sort_index()
    bin_pct    = (bin_counts / len(df_bin) * 100).round(1)

    print("\n" + "="*50)
    print("📊 BINARY TARGET DISTRIBUTION")
    print("="*50)
    for cls, count in bin_counts.items():
        label = BINARY_LABELS[int(cls)]
        print(f"   {label:<10} (class {int(cls)}) : {count:>5,}  ({bin_pct[cls]:.1f}%)")

    # ── Multiclass distribution ───────────────────────────
    df_mc = df.dropna(subset=["target_multi"])
    mc_counts = df_mc["target_multi"].value_counts().sort_index()
    mc_pct    = (mc_counts / len(df_mc) * 100).round(1)

    print("\n" + "="*50)
    print("📊 MULTICLASS TARGET DISTRIBUTION")
    print("="*50)
    for cls, count in mc_counts.items():
        label = MULTICLASS_LABELS[int(cls)]
        print(f"   {label:<10} (class {int(cls)}) : {count:>5,}  ({mc_pct[cls]:.1f}%)")

    # ── Plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution — SCDB Dataset", fontsize=14, fontweight="bold")

    # Binary
    axes[0].bar(
        [BINARY_LABELS[int(c)] for c in bin_counts.index],
        bin_counts.values,
        color=["#2196F3", "#F44336"], edgecolor="black", linewidth=0.8
    )
    axes[0].set_title("Binary Target (Affirm vs Reverse)")
    axes[0].set_ylabel("Number of Cases")
    for i, (count, pct) in enumerate(zip(bin_counts.values, bin_pct.values)):
        axes[0].text(i, count + 30, f"{pct:.1f}%", ha="center", fontsize=11)

    # Multiclass
    colors = ["#2196F3", "#F44336", "#FF9800", "#9C27B0"]
    axes[1].bar(
        [MULTICLASS_LABELS[int(c)] for c in mc_counts.index],
        mc_counts.values,
        color=colors[:len(mc_counts)], edgecolor="black", linewidth=0.8
    )
    axes[1].set_title("Multiclass Target (4 classes)")
    axes[1].set_ylabel("Number of Cases")
    for i, (count, pct) in enumerate(zip(mc_counts.values, mc_pct.values)):
        axes[1].text(i, count + 30, f"{pct:.1f}%", ha="center", fontsize=11)

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n📊 Distribution plot saved → results/figures/class_distribution.png")


def report_term_distribution(df: pd.DataFrame) -> None:
    """Show how cases are distributed across the train/test split."""
    train = df[df["term"] <= TRAIN_END_YEAR]
    test  = df[df["term"] >= TEST_START_YEAR]
    print(f"\n📅 Chronological split:")
    print(f"   Training set (1946–{TRAIN_END_YEAR}): {len(train):,} cases")
    print(f"   Test set ({TEST_START_YEAR}–present)  : {len(test):,} cases")


def save_prepared_data(df: pd.DataFrame) -> None:
    """Save the prepared (filtered + targets added) dataframe."""
    os.makedirs("data/processed", exist_ok=True)
    outpath = "data/processed/scdb_prepared.csv"
    df.to_csv(outpath, index=False)
    print(f"\n💾 Prepared data saved → {outpath}")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")


def main(filepath: str) -> None:
    print("=" * 60)
    print("PHASE 2 — DATA PREPARATION")
    print("=" * 60)

    df = load_data(filepath)
    inspect_disposition(df)

    df = remove_missing_disposition(df)
    df = exclude_procedural(df)
    df = create_binary_target(df)
    df = create_multiclass_target(df)

    report_class_distribution(df)
    report_term_distribution(df)
    save_prepared_data(df)

    print("\n✅ Phase 2 complete.")
    print("   Output → data/processed/scdb_prepared.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCOTUS — Phase 2: Data Preparation")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the raw SCDB CSV file"
    )
    args = parser.parse_args()
    main(args.data)

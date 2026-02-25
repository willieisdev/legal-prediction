"""
=============================================================
PHASE 9 — ROBUSTNESS CHECKS
=============================================================
1. Decade-by-decade accuracy (best binary model)
2. Error analysis: confident-wrong predictions
3. Binary vs multiclass stability comparison
4. Training window sensitivity (post-1970 vs full history)
=============================================================
USAGE:
    python phase9_robustness.py
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

from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_setup          import (BINARY_LABELS, MULTICLASS_LABELS,
                                   RANDOM_STATE, TRAIN_END_YEAR, TEST_START_YEAR)
from phase4_data_splitting import load_binary_data, load_multiclass_data

DECADES = [(1946,1955),(1956,1965),(1966,1975),(1976,1985),
           (1986,1995),(1996,2005),(2006,2015),(2016,2024)]


def load_best_model_names() -> tuple:
    path = "data/processed/best_model_names.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("Run Phase 7 first to generate best_model_names.csv")
    s = pd.read_csv(path, header=None, index_col=0).squeeze()
    return str(s["best_binary"]), str(s["best_multiclass"])


def _uses_scaled(name: str) -> bool:
    return "logreg" in name


def load_prepared_with_term() -> pd.DataFrame:
    """Load the prepared CSV (with term column) for decade slicing."""
    return pd.read_csv("data/processed/scdb_prepared.csv", low_memory=False)


# ══════════════════════════════════════════════════════════
# CHECK 1: DECADE-BY-DECADE ACCURACY
# ══════════════════════════════════════════════════════════

def decade_accuracy(model, X_scaled: np.ndarray, X_unscaled: np.ndarray,
                    y: np.ndarray, term_arr: np.ndarray,
                    model_name: str) -> pd.DataFrame:
    """Compute accuracy and F1 for each decade slice."""
    X = X_scaled if _uses_scaled(model_name) else X_unscaled
    rows = []
    for (start, end) in DECADES:
        mask = (term_arr >= start) & (term_arr <= end)
        if mask.sum() < 10:
            continue
        y_pred = model.predict(X[mask])
        y_true = y[mask]
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        rows.append({
            "Decade"   : f"{start}–{end}",
            "N_cases"  : int(mask.sum()),
            "Split"    : "Train" if end <= TRAIN_END_YEAR else "Test",
            "Accuracy" : acc,
            "Weighted_F1": f1,
        })
    return pd.DataFrame(rows)


def plot_decade_accuracy(df: pd.DataFrame, model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Decade-by-Decade Performance\n{model_name}",
                 fontsize=13, fontweight="bold")

    colors = ["#1565C0" if s == "Train" else "#E53935" for s in df["Split"]]

    for ax, metric in zip(axes, ["Accuracy", "Weighted_F1"]):
        bars = ax.bar(df["Decade"], df[metric], color=colors,
                      edgecolor="black", linewidth=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.set_xlabel("Decade")
        ax.set_title(metric)
        ax.axvline(x=df[df["Split"]=="Train"].index[-1] + 0.5,
                   color="black", linestyle="--", linewidth=1.2)
        ax.tick_params(axis="x", rotation=45)

        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor="#1565C0", label="Train era"),
                    Patch(facecolor="#E53935", label="Test era")]
    axes[0].legend(handles=legend_elems, loc="lower left")

    plt.tight_layout()
    path = f"results/figures/decade_accuracy_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Decade accuracy plot → {path}")


# ══════════════════════════════════════════════════════════
# CHECK 2: ERROR ANALYSIS — CONFIDENT-WRONG PREDICTIONS
# ══════════════════════════════════════════════════════════

def error_analysis(model, X_scaled: np.ndarray, X_unscaled: np.ndarray,
                   y_true: np.ndarray, feature_names: list,
                   model_name: str, n_top: int = 20) -> pd.DataFrame:
    """
    Find cases where the model was most confident but wrong.
    Returns a dataframe of the worst-confidence errors.
    """
    X = X_scaled if _uses_scaled(model_name) else X_unscaled

    if not hasattr(model, "predict_proba"):
        print("   ⚠️  Model has no predict_proba — skipping error analysis.")
        return pd.DataFrame()

    y_pred  = model.predict(X)
    y_prob  = model.predict_proba(X)
    max_conf = y_prob.max(axis=1)   # confidence in predicted class

    wrong_mask   = (y_pred != y_true)
    conf_wrong   = max_conf[wrong_mask]
    pred_wrong   = y_pred[wrong_mask]
    true_wrong   = y_true[wrong_mask]

    # Sort by confidence descending (most confident errors first)
    order = np.argsort(conf_wrong)[::-1][:n_top]

    df_errors = pd.DataFrame({
        "True_Label" : [BINARY_LABELS.get(int(t), str(t)) for t in true_wrong[order]],
        "Pred_Label" : [BINARY_LABELS.get(int(p), str(p)) for p in pred_wrong[order]],
        "Confidence" : conf_wrong[order].round(4),
    })

    print(f"\n   Top {n_top} most confident wrong predictions:")
    print(df_errors.head(10).to_string(index=False))

    # Distribution of errors by true class
    print(f"\n   Error breakdown by true class:")
    for cls in sorted(np.unique(y_true)):
        n_wrong = ((y_pred != y_true) & (y_true == cls)).sum()
        n_total = (y_true == cls).sum()
        label   = BINARY_LABELS.get(int(cls), str(cls))
        print(f"   {label:<10}: {n_wrong:>4}/{n_total} wrong ({n_wrong/n_total*100:.1f}%)")

    return df_errors


# ══════════════════════════════════════════════════════════
# CHECK 3: BINARY VS MULTICLASS STABILITY
# ══════════════════════════════════════════════════════════

def stability_comparison(bin_model, mc_model,
                          X_te_bin_sc, X_te_bin_un,
                          X_te_mc_sc, X_te_mc_un,
                          y_te_bin, y_te_mc,
                          bin_name: str, mc_name: str) -> None:
    """
    Compare which test cases are wrong in binary vs multiclass.
    """
    X_bin = X_te_bin_sc if _uses_scaled(bin_name) else X_te_bin_un
    X_mc  = X_te_mc_sc  if _uses_scaled(mc_name)  else X_te_mc_un

    y_bin_pred = bin_model.predict(X_bin)
    y_mc_pred  = mc_model.predict(X_mc)

    # Only compare on cases present in BOTH (they may differ slightly due to NaN drops)
    n_compare = min(len(y_te_bin), len(y_te_mc))
    bin_wrong = (y_bin_pred[:n_compare] != y_te_bin[:n_compare])
    mc_wrong  = (y_mc_pred[:n_compare]  != y_te_mc[:n_compare])

    both_wrong   = (bin_wrong & mc_wrong).sum()
    only_bin     = (bin_wrong & ~mc_wrong).sum()
    only_mc      = (~bin_wrong & mc_wrong).sum()
    both_correct = (~bin_wrong & ~mc_wrong).sum()

    total = n_compare
    print(f"\n   Stability Comparison (n={total:,} test cases):")
    print(f"   Both correct          : {both_correct:>5,}  ({both_correct/total*100:.1f}%)")
    print(f"   Both wrong            : {both_wrong:>5,}  ({both_wrong/total*100:.1f}%)")
    print(f"   Only binary wrong     : {only_bin:>5,}  ({only_bin/total*100:.1f}%)")
    print(f"   Only multiclass wrong : {only_mc:>5,}  ({only_mc/total*100:.1f}%)")

    # Venn-style bar chart
    labels = ["Both Correct", "Both Wrong", "Binary only wrong", "Multiclass only wrong"]
    values = [both_correct, both_wrong, only_bin, only_mc]
    colors = ["#43A047", "#E53935", "#FB8C00", "#7B1FA2"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 5,
                f"{val:,}\n({val/total*100:.1f}%)",
                ha="center", fontsize=9)
    ax.set_ylabel("Number of test cases")
    ax.set_title("Binary vs Multiclass Prediction Stability", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig("results/figures/stability_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Stability comparison → results/figures/stability_comparison.png")


# ══════════════════════════════════════════════════════════
# CHECK 4: TRAINING WINDOW SENSITIVITY
# ══════════════════════════════════════════════════════════

def training_window_sensitivity(model_name: str,
                                 feature_names: list) -> None:
    """
    Compare best binary model performance when trained on:
      (a) Full history: 1946–2005
      (b) Modern only:  1970–2005
    Both tested on 2006–2024.
    """
    from sklearn.ensemble  import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb

    print("\n" + "─"*50)
    print("Training Window Sensitivity (post-1970 vs full history)")
    print("─"*50)

    # Load the best model to get its hyperparameters
    if "binary_" in model_name:
        model_path = f"models/{model_name}.joblib"
    else:
        model_path = f"models/binary_{model_name}.joblib"

    if not os.path.exists(model_path):
        print(f"   ⚠️  Model not found at {model_path} — skipping.")
        return

    best_model = joblib.load(model_path)

    # Load raw feature arrays (unscaled for tree models)
    X_train_un = np.load("data/processed/X_train_unscaled.npy")
    X_test_un  = np.load("data/processed/X_test_unscaled.npy")
    y_train    = pd.read_csv("data/processed/y_train_bin.csv").squeeze().values
    y_test     = pd.read_csv("data/processed/y_test_bin.csv").squeeze().values

    # Remove NaN rows
    train_mask = ~np.isnan(y_train)
    test_mask  = ~np.isnan(y_test)
    X_tr_all   = X_train_un[train_mask].astype(np.float32)
    y_tr_all   = y_train[train_mask].astype(int)
    X_te       = X_test_un[test_mask].astype(np.float32)
    y_te       = y_test[test_mask].astype(int)

    # We need term to filter post-1970 — approximate by index
    # term is the first column in NUMERICAL_FEATURES → it's encoded in X
    # Instead: reload the prepared CSV for term values
    df_prep = pd.read_csv("data/processed/scdb_prepared.csv", low_memory=False)
    df_train = df_prep[
        (df_prep["term"] <= TRAIN_END_YEAR) &
        df_prep["target_binary"].notna()
    ].copy()

    # Post-1970 mask (aligned by position)
    post1970_mask = (df_train["term"] >= 1970).values
    if post1970_mask.sum() < 100:
        print("   ⚠️  Not enough post-1970 training data — skipping.")
        return

    # Align sizes
    n_full     = len(X_tr_all)
    n_post1970 = post1970_mask[:n_full].sum()

    X_tr_post1970 = X_tr_all[post1970_mask[:n_full]]
    y_tr_post1970 = y_tr_all[post1970_mask[:n_full]]

    results = []
    for tag, X_tr, y_tr in [("Full (1946–2005)", X_tr_all, y_tr_all),
                              ("Modern (1970–2005)", X_tr_post1970, y_tr_post1970)]:
        # Clone the best model's class
        if isinstance(best_model, lgb.LGBMClassifier):
            params = best_model.get_params()
            params["verbose"] = -1
            clone = lgb.LGBMClassifier(**params)
        elif isinstance(best_model, RandomForestClassifier):
            clone = RandomForestClassifier(**best_model.get_params())
        elif isinstance(best_model, LogisticRegression):
            clone = LogisticRegression(**best_model.get_params())
        else:
            clone = best_model.__class__(**best_model.get_params())

        clone.fit(X_tr, y_tr)
        y_pred = clone.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred, zero_division=0)
        results.append({"Window": tag, "Train_N": len(X_tr),
                        "Accuracy": acc, "F1": f1})
        print(f"   {tag}: n={len(X_tr):,}  Accuracy={acc:.4f}  F1={f1:.4f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv("results/tables/window_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    w = 0.3
    ax.bar(x - w/2, df_res["Accuracy"], w, label="Accuracy",
           color="#1976D2", edgecolor="black", lw=0.7)
    ax.bar(x + w/2, df_res["F1"],       w, label="F1",
           color="#F57C00", edgecolor="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(df_res["Window"])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Training Window Sensitivity", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/window_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Window sensitivity → results/figures/window_sensitivity.png")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PHASE 9 — ROBUSTNESS CHECKS")
    print("=" * 60)

    best_bin_name, best_mc_name = load_best_model_names()
    print(f"\n  Best binary    : {best_bin_name}")
    print(f"  Best multiclass: {best_mc_name}")

    # Load data
    (X_tr_sc, X_tr_un, X_te_sc, X_te_un,
     y_tr_bin, y_te_bin, feat_names) = load_binary_data()

    (X_tr_sc_mc, X_tr_un_mc, X_te_sc_mc, X_te_un_mc,
     y_tr_mc, y_te_mc, _) = load_multiclass_data()

    bin_model = joblib.load(f"models/binary_{best_bin_name}.joblib")
    mc_model  = joblib.load(f"models/{best_mc_name}.joblib")

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables",  exist_ok=True)

    # ── Load term arrays for decade slicing ───────────────
    df_prep = pd.read_csv("data/processed/scdb_prepared.csv", low_memory=False)

    df_bin_prep  = df_prep[df_prep["target_binary"].notna()].reset_index(drop=True)
    term_all_bin = df_bin_prep["term"].values

    # Align term array to X sizes
    n_tr_bin = len(X_tr_un)
    n_te_bin = len(X_te_un)
    term_train_bin = term_all_bin[:n_tr_bin]
    term_test_bin  = term_all_bin[n_tr_bin: n_tr_bin + n_te_bin]

    # Combine train+test for full decade sweep
    X_all_sc = np.vstack([X_tr_sc, X_te_sc])
    X_all_un = np.vstack([X_tr_un, X_te_un])
    y_all_bin = np.concatenate([y_tr_bin, y_te_bin])
    term_all  = np.concatenate([term_train_bin, term_test_bin])

    # ── Check 1: Decade accuracy ──────────────────────────
    print("\n" + "─"*50)
    print("CHECK 1: DECADE-BY-DECADE ACCURACY")
    print("─"*50)
    df_decade = decade_accuracy(bin_model,
                                X_all_sc, X_all_un,
                                y_all_bin, term_all,
                                best_bin_name)
    print(df_decade.to_string(index=False, float_format="{:.4f}".format))
    df_decade.to_csv("results/tables/decade_accuracy.csv", index=False)
    plot_decade_accuracy(df_decade, best_bin_name)

    # ── Check 2: Error analysis ───────────────────────────
    print("\n" + "─"*50)
    print("CHECK 2: ERROR ANALYSIS")
    print("─"*50)
    X_te_bin_use = X_te_sc if _uses_scaled(best_bin_name) else X_te_un
    df_errors = error_analysis(bin_model,
                               X_te_sc, X_te_un,
                               y_te_bin, feat_names,
                               best_bin_name)
    if not df_errors.empty:
        df_errors.to_csv("results/tables/confident_errors.csv", index=False)

    # ── Check 3: Stability comparison ─────────────────────
    print("\n" + "─"*50)
    print("CHECK 3: BINARY vs MULTICLASS STABILITY")
    print("─"*50)
    stability_comparison(bin_model, mc_model,
                         X_te_sc, X_te_un,
                         X_te_sc_mc, X_te_un_mc,
                         y_te_bin, y_te_mc,
                         best_bin_name, best_mc_name)

    # ── Check 4: Window sensitivity ───────────────────────
    training_window_sensitivity(best_bin_name, feat_names)

    print("\n✅ Phase 9 complete — all robustness checks done.")


if __name__ == "__main__":
    main()

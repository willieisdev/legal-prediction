"""
=============================================================
PHASE 7 — EVALUATION (TEST SET)
=============================================================
Binary task — per-class metrics for EACH model × strategy:
  Overall : Accuracy, ROC-AUC
  Class 0 (Affirm)  : Precision, Recall, F1, Support
  Class 1 (Reverse) : Precision, Recall, F1, Support
  Averaged: Macro-Precision, Macro-Recall, Macro-F1
            Weighted-Precision, Weighted-Recall, Weighted-F1

Multiclass task — per-class metrics for EACH model × strategy:
  Overall : Accuracy, ROC-AUC (OvR)
  Class 0 (Affirm)  : Precision, Recall, F1, Support
  Class 1 (Reverse) : Precision, Recall, F1, Support
  Class 2 (Remand)  : Precision, Recall, F1, Support
  Class 3 (Other)   : Precision, Recall, F1, Support
  Averaged: Macro-Precision, Macro-Recall, Macro-F1
            Weighted-Precision, Weighted-Recall, Weighted-F1

Outputs:
  results/tables/binary_results_overall.csv
  results/tables/binary_results_perclass.csv
  results/tables/multiclass_results_overall.csv
  results/tables/multiclass_results_perclass.csv
  results/figures/binary_confusion_<model>.png       (one per model)
  results/figures/multiclass_confusion_<model>.png   (one per model)
  results/figures/roc_curves_binary.png
  results/figures/perclass_f1_binary.png
  results/figures/perclass_f1_multiclass.png
  results/figures/model_comparison.png
=============================================================
USAGE:
    python phase7_evaluation.py
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

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay
)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_setup          import BINARY_LABELS, MULTICLASS_LABELS, RANDOM_STATE
from phase4_data_splitting import load_binary_data, load_multiclass_data

STRATEGY_LABELS = {
    "unbalanced": "Unbalanced",
    "smote"     : "SMOTE",
    "weighted"  : "Class Weights",
}


def _uses_scaled(name: str) -> bool:
    return "logreg" in name


def _get_X(name: str, X_sc: np.ndarray, X_un: np.ndarray) -> np.ndarray:
    return X_sc if _uses_scaled(name) else X_un


def _parse_strategy(name: str) -> str:
    for tag in ["smote", "weighted"]:
        if name.endswith(tag):
            return STRATEGY_LABELS[tag]
    return STRATEGY_LABELS["unbalanced"]


def _roc_auc(model, X_test: np.ndarray,
             y_test: np.ndarray, multi_class: str = "raise") -> float:
    """Safely compute ROC-AUC; returns NaN if not possible."""
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if multi_class == "ovr":
                return roc_auc_score(y_test, proba,
                                     multi_class="ovr", average="macro")
            else:
                return roc_auc_score(y_test, proba[:, 1])
        elif hasattr(model, "decision_function"):
            score = model.decision_function(X_test)
            return roc_auc_score(y_test, score)
    except Exception:
        pass
    return np.nan


def _separator(title: str) -> None:
    print("\n" + "═"*60)
    print(f"  {title}")
    print("═"*60)


# ══════════════════════════════════════════════════════════
# PER-CLASS METRICS EXTRACTOR
# ══════════════════════════════════════════════════════════

def per_class_metrics_binary(y_true: np.ndarray,
                              y_pred: np.ndarray) -> dict:
    """
    Returns a flat dict of per-class + averaged metrics for binary task.
    Classes: 0=Affirm, 1=Reverse
    """
    labels  = [0, 1]
    names   = [BINARY_LABELS[0], BINARY_LABELS[1]]

    prec_per   = precision_score(y_true, y_pred, labels=labels,
                                 average=None, zero_division=0)
    rec_per    = recall_score(y_true, y_pred, labels=labels,
                              average=None, zero_division=0)
    f1_per     = f1_score(y_true, y_pred, labels=labels,
                          average=None, zero_division=0)
    support    = [int((y_true == lbl).sum()) for lbl in labels]

    metrics = {}

    # ── Per-class ─────────────────────────────────────────
    for i, cls_name in enumerate(names):
        metrics[f"Precision_{cls_name}"] = round(prec_per[i], 4)
        metrics[f"Recall_{cls_name}"]    = round(rec_per[i],  4)
        metrics[f"F1_{cls_name}"]        = round(f1_per[i],   4)
        metrics[f"Support_{cls_name}"]   = support[i]

    # ── Averaged ──────────────────────────────────────────
    for avg in ["macro", "weighted"]:
        tag = avg.capitalize()
        metrics[f"Precision_{tag}"] = round(
            precision_score(y_true, y_pred, average=avg, zero_division=0), 4)
        metrics[f"Recall_{tag}"]    = round(
            recall_score(y_true, y_pred, average=avg, zero_division=0), 4)
        metrics[f"F1_{tag}"]        = round(
            f1_score(y_true, y_pred, average=avg, zero_division=0), 4)

    return metrics


def per_class_metrics_multiclass(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  present_classes: list) -> dict:
    """
    Returns a flat dict of per-class + averaged metrics for multiclass task.
    """
    prec_per = precision_score(y_true, y_pred, labels=present_classes,
                               average=None, zero_division=0)
    rec_per  = recall_score(y_true, y_pred, labels=present_classes,
                            average=None, zero_division=0)
    f1_per   = f1_score(y_true, y_pred, labels=present_classes,
                        average=None, zero_division=0)
    support  = [int((y_true == c).sum()) for c in present_classes]

    metrics = {}

    # ── Per-class ─────────────────────────────────────────
    for i, cls_idx in enumerate(present_classes):
        cls_name = MULTICLASS_LABELS.get(cls_idx, f"Class_{cls_idx}")
        metrics[f"Precision_{cls_name}"] = round(prec_per[i], 4)
        metrics[f"Recall_{cls_name}"]    = round(rec_per[i],  4)
        metrics[f"F1_{cls_name}"]        = round(f1_per[i],   4)
        metrics[f"Support_{cls_name}"]   = support[i]

    # ── Averaged ──────────────────────────────────────────
    for avg in ["macro", "weighted"]:
        tag = avg.capitalize()
        metrics[f"Precision_{tag}"] = round(
            precision_score(y_true, y_pred, average=avg, zero_division=0), 4)
        metrics[f"Recall_{tag}"]    = round(
            recall_score(y_true, y_pred, average=avg, zero_division=0), 4)
        metrics[f"F1_{tag}"]        = round(
            f1_score(y_true, y_pred, average=avg, zero_division=0), 4)

    return metrics


# ══════════════════════════════════════════════════════════
# CONSOLE PRINTERS
# ══════════════════════════════════════════════════════════

def print_binary_perclass(model_name: str, strategy: str,
                           overall: dict, perclass: dict) -> None:
    print(f"\n  ┌─ [{model_name}]  Strategy: {strategy}")
    print(f"  │  Overall  →  "
          f"Accuracy={overall['Accuracy']:.4f}  "
          f"ROC-AUC={overall['ROC_AUC']:.4f}")
    print(f"  │")
    print(f"  │  {'Class':<12} {'Precision':>10} {'Recall':>10} "
          f"{'F1':>10} {'Support':>10}")
    print(f"  │  {'─'*54}")

    for cls_name in [BINARY_LABELS[0], BINARY_LABELS[1]]:
        print(f"  │  {cls_name:<12} "
              f"{perclass[f'Precision_{cls_name}']:>10.4f} "
              f"{perclass[f'Recall_{cls_name}']:>10.4f} "
              f"{perclass[f'F1_{cls_name}']:>10.4f} "
              f"{perclass[f'Support_{cls_name}']:>10,}")

    print(f"  │  {'─'*54}")
    print(f"  │  {'Macro avg':<12} "
          f"{perclass['Precision_Macro']:>10.4f} "
          f"{perclass['Recall_Macro']:>10.4f} "
          f"{perclass['F1_Macro']:>10.4f}")
    print(f"  │  {'Weighted avg':<12} "
          f"{perclass['Precision_Weighted']:>10.4f} "
          f"{perclass['Recall_Weighted']:>10.4f} "
          f"{perclass['F1_Weighted']:>10.4f}")
    print(f"  └{'─'*55}")


def print_multiclass_perclass(model_name: str, strategy: str,
                               overall: dict, perclass: dict,
                               present_classes: list) -> None:
    print(f"\n  ┌─ [{model_name}]  Strategy: {strategy}")
    print(f"  │  Overall  →  "
          f"Accuracy={overall['Accuracy']:.4f}  "
          f"ROC-AUC(OvR)={overall['ROC_AUC_OvR']:.4f}")
    print(f"  │")
    print(f"  │  {'Class':<12} {'Precision':>10} {'Recall':>10} "
          f"{'F1':>10} {'Support':>10}")
    print(f"  │  {'─'*54}")

    for cls_idx in present_classes:
        cls_name = MULTICLASS_LABELS.get(cls_idx, f"Class_{cls_idx}")
        print(f"  │  {cls_name:<12} "
              f"{perclass[f'Precision_{cls_name}']:>10.4f} "
              f"{perclass[f'Recall_{cls_name}']:>10.4f} "
              f"{perclass[f'F1_{cls_name}']:>10.4f} "
              f"{perclass[f'Support_{cls_name}']:>10,}")

    print(f"  │  {'─'*54}")
    print(f"  │  {'Macro avg':<12} "
          f"{perclass['Precision_Macro']:>10.4f} "
          f"{perclass['Recall_Macro']:>10.4f} "
          f"{perclass['F1_Macro']:>10.4f}")
    print(f"  │  {'Weighted avg':<12} "
          f"{perclass['Precision_Weighted']:>10.4f} "
          f"{perclass['Recall_Weighted']:>10.4f} "
          f"{perclass['F1_Weighted']:>10.4f}")
    print(f"  └{'─'*55}")


# ══════════════════════════════════════════════════════════
# CONFUSION MATRIX PLOTS
# ══════════════════════════════════════════════════════════

def plot_confusion_binary(y_true, y_pred, model_name: str) -> None:
    labels = [BINARY_LABELS[0], BINARY_LABELS[1]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Add row/col totals
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=11, fontweight="bold")

    # Counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.8, ax=axes[0], cbar=False)
    axes[0].set_xlabel("Predicted");  axes[0].set_ylabel("Actual")
    axes[0].set_title("Counts")

    # Row-normalised %
    annot = np.array([[f"{v:.1f}%" for v in row] for row in cm_pct])
    sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.8, ax=axes[1], cbar=False,
                vmin=0, vmax=100)
    axes[1].set_xlabel("Predicted");  axes[1].set_ylabel("Actual")
    axes[1].set_title("Row-normalised (%)")

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(f"results/figures/binary_confusion_{model_name}.png",
                dpi=130, bbox_inches="tight")
    plt.close()


def plot_confusion_multiclass(y_true, y_pred,
                               class_labels: list,
                               present_classes: list,
                               model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_pct = np.where(cm.sum(axis=1, keepdims=True) > 0,
                          cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100,
                          0.0)

    n = len(class_labels)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, n*3), max(4, n+1)))
    fig.suptitle(f"Confusion Matrix (Multiclass) — {model_name}",
                 fontsize=11, fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=class_labels, yticklabels=class_labels,
                linewidths=0.8, ax=axes[0], cbar=False)
    axes[0].set_xlabel("Predicted");  axes[0].set_ylabel("Actual")
    axes[0].set_title("Counts")

    annot = np.array([[f"{v:.1f}%" for v in row] for row in cm_pct])
    sns.heatmap(cm_pct, annot=annot, fmt="", cmap="YlOrRd",
                xticklabels=class_labels, yticklabels=class_labels,
                linewidths=0.8, ax=axes[1], cbar=False,
                vmin=0, vmax=100)
    axes[1].set_xlabel("Predicted");  axes[1].set_ylabel("Actual")
    axes[1].set_title("Row-normalised (%)")

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(f"results/figures/multiclass_confusion_{model_name}.png",
                dpi=130, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════
# PER-CLASS F1 COMPARISON PLOTS
# ══════════════════════════════════════════════════════════

def plot_perclass_f1_binary(df_perclass: pd.DataFrame) -> None:
    """Grouped bar chart: per-class F1 for every model."""
    cls_names = [BINARY_LABELS[0], BINARY_LABELS[1]]
    f1_cols   = [f"F1_{c}" for c in cls_names]

    models = df_perclass["Model"].tolist()
    x      = np.arange(len(models))
    width  = 0.30
    colors = ["#1976D2", "#D32F2F", "#388E3C"]

    fig, ax = plt.subplots(figsize=(max(10, len(models)*1.2), 6))
    for i, (col, cls_name) in enumerate(zip(f1_cols, cls_names)):
        offset = (i - len(f1_cols)/2 + 0.5) * width
        bars = ax.bar(x + offset, df_perclass[col].values,
                      width, label=f"F1 — {cls_name}",
                      color=colors[i], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, df_perclass[col].values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score — Binary Models (Test Set)",
                 fontweight="bold", fontsize=12)
    ax.legend(loc="upper right")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    plt.tight_layout()
    plt.savefig("results/figures/perclass_f1_binary.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Per-class F1 chart → results/figures/perclass_f1_binary.png")


def plot_perclass_f1_multiclass(df_perclass: pd.DataFrame,
                                 present_classes: list) -> None:
    """Grouped bar chart: per-class F1 for every multiclass model."""
    cls_names = [MULTICLASS_LABELS.get(c, str(c)) for c in present_classes]
    f1_cols   = [f"F1_{c}" for c in cls_names]
    f1_cols   = [c for c in f1_cols if c in df_perclass.columns]

    models = df_perclass["Model"].tolist()
    x      = np.arange(len(models))
    width  = 0.20
    colors = ["#1976D2", "#D32F2F", "#F57C00", "#7B1FA2",
              "#388E3C", "#00838F"]

    fig, ax = plt.subplots(figsize=(max(12, len(models)*1.5), 6))
    for i, col in enumerate(f1_cols):
        cls_label = col.replace("F1_", "")
        offset = (i - len(f1_cols)/2 + 0.5) * width
        bars = ax.bar(x + offset, df_perclass[col].values,
                      width, label=f"F1 — {cls_label}",
                      color=colors[i % len(colors)],
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, df_perclass[col].values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score — Multiclass Models (Test Set)",
                 fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig("results/figures/perclass_f1_multiclass.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Per-class F1 chart → results/figures/perclass_f1_multiclass.png")


# ══════════════════════════════════════════════════════════
# ROC CURVES
# ══════════════════════════════════════════════════════════

def plot_roc_curves(roc_data: dict) -> None:
    if not roc_data:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))
    for (name, (y_true, y_prob)), color in zip(roc_data.items(), colors):
        try:
            RocCurveDisplay.from_predictions(
                y_true, y_prob, name=name, ax=ax, color=color)
        except Exception:
            pass
    ax.plot([0,1],[0,1],"k--", linewidth=1, label="Random (AUC=0.50)")
    ax.set_title("ROC Curves — Binary Models (Test Set)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig("results/figures/roc_curves_binary.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 ROC curves → results/figures/roc_curves_binary.png")


# ══════════════════════════════════════════════════════════
# BINARY EVALUATION
# ══════════════════════════════════════════════════════════

def evaluate_binary() -> tuple:
    _separator("BINARY EVALUATION  (Test Set)")

    (X_tr_sc, X_tr_un, X_te_sc, X_te_un,
     y_train, y_test, feature_names) = load_binary_data()

    model_names = pd.read_csv(
        "data/processed/binary_model_names.csv").squeeze().tolist()

    overall_rows  = []
    perclass_rows = []
    roc_data      = {}

    for name in model_names:
        path = f"models/binary_{name}.joblib"
        if not os.path.exists(path):
            print(f"   ⚠️  Not found: {path} — skipping.")
            continue

        model    = joblib.load(path)
        X_test   = _get_X(name, X_te_sc, X_te_un)
        y_pred   = model.predict(X_test)
        strategy = _parse_strategy(name)

        # Overall metrics
        auc = _roc_auc(model, X_test, y_test)
        if hasattr(model, "predict_proba") and not np.isnan(auc):
            roc_data[name] = (y_test, model.predict_proba(X_test)[:, 1])

        overall = {
            "Model"    : name,
            "Strategy" : strategy,
            "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
            "ROC_AUC"  : round(auc, 4) if not np.isnan(auc) else np.nan,
        }
        overall_rows.append(overall)

        # Per-class metrics
        pc = per_class_metrics_binary(y_test, y_pred)
        perclass_rows.append({"Model": name, "Strategy": strategy, **pc})

        # Print full breakdown to console
        print_binary_perclass(name, strategy, overall, pc)

        # Confusion matrix (counts + row-%)
        plot_confusion_binary(y_test, y_pred, name)

    df_overall  = pd.DataFrame(overall_rows).sort_values(
        "ROC_AUC", ascending=False).reset_index(drop=True)
    df_perclass = pd.DataFrame(perclass_rows)
    # Align perclass order to overall order
    df_perclass = df_perclass.set_index("Model").reindex(
        df_overall["Model"]).reset_index()

    # Save CSVs
    os.makedirs("results/tables", exist_ok=True)
    df_overall.to_csv("results/tables/binary_results_overall.csv",  index=False)
    df_perclass.to_csv("results/tables/binary_results_perclass.csv", index=False)
    print(f"\n💾 Saved → results/tables/binary_results_overall.csv")
    print(f"💾 Saved → results/tables/binary_results_perclass.csv")

    # Plots
    plot_roc_curves(roc_data)
    plot_perclass_f1_binary(df_perclass)

    # ── Summary table ─────────────────────────────────────
    print("\n" + "═"*60)
    print("  BINARY — OVERALL SUMMARY (sorted by ROC-AUC)")
    print("═"*60)
    print(df_overall.to_string(index=False, float_format="{:.4f}".format))

    print("\n" + "═"*60)
    print("  BINARY — PER-CLASS SUMMARY")
    print("═"*60)
    # Print only the key per-class columns cleanly
    display_cols = (["Model", "Strategy"] +
                    [f"{m}_{c}"
                     for c in [BINARY_LABELS[0], BINARY_LABELS[1],
                                "Macro", "Weighted"]
                     for m in ["Precision", "Recall", "F1"]
                     if f"{m}_{c}" in df_perclass.columns])
    display_cols = [c for c in display_cols if c in df_perclass.columns]
    print(df_perclass[display_cols].to_string(
        index=False, float_format="{:.4f}".format))

    return df_overall, df_perclass


# ══════════════════════════════════════════════════════════
# MULTICLASS EVALUATION
# ══════════════════════════════════════════════════════════

def evaluate_multiclass() -> tuple:
    _separator("MULTICLASS EVALUATION  (Test Set)")

    (X_tr_sc, X_tr_un, X_te_sc, X_te_un,
     y_train, y_test, feature_names) = load_multiclass_data()

    model_names = pd.read_csv(
        "data/processed/multiclass_model_names.csv").squeeze().tolist()

    # Determine all classes present in test set
    all_classes = sorted(np.unique(y_test).astype(int))
    cls_labels  = [MULTICLASS_LABELS.get(c, str(c)) for c in all_classes]

    overall_rows  = []
    perclass_rows = []

    for name in model_names:
        path = f"models/{name}.joblib"
        if not os.path.exists(path):
            print(f"   ⚠️  Not found: {path} — skipping.")
            continue

        model    = joblib.load(path)
        X_test   = _get_X(name, X_te_sc, X_te_un)
        y_pred   = model.predict(X_test).astype(int)
        strategy = _parse_strategy(name)

        # Ensure predicted classes aligned to all_classes
        present = sorted(
            np.unique(np.concatenate([y_test, y_pred])).astype(int))
        present_labels = [MULTICLASS_LABELS.get(c, str(c)) for c in present]

        # Overall metrics
        auc_ovr = _roc_auc(model, X_test, y_test, multi_class="ovr")

        overall = {
            "Model"       : name,
            "Strategy"    : strategy,
            "Accuracy"    : round(accuracy_score(y_test, y_pred), 4),
            "ROC_AUC_OvR" : round(auc_ovr, 4) if not np.isnan(auc_ovr) else np.nan,
        }
        overall_rows.append(overall)

        # Per-class metrics
        pc = per_class_metrics_multiclass(y_test, y_pred, present)
        perclass_rows.append({"Model": name, "Strategy": strategy, **pc})

        # Print full breakdown to console
        print_multiclass_perclass(name, strategy, overall, pc, present)

        # Confusion matrix
        plot_confusion_multiclass(y_test, y_pred,
                                   present_labels, present, name)

    df_overall  = pd.DataFrame(overall_rows).sort_values(
        "ROC_AUC_OvR", ascending=False).reset_index(drop=True)
    df_perclass = pd.DataFrame(perclass_rows)
    df_perclass = df_perclass.set_index("Model").reindex(
        df_overall["Model"]).reset_index()

    # Save CSVs
    df_overall.to_csv("results/tables/multiclass_results_overall.csv",  index=False)
    df_perclass.to_csv("results/tables/multiclass_results_perclass.csv", index=False)
    print(f"\n💾 Saved → results/tables/multiclass_results_overall.csv")
    print(f"💾 Saved → results/tables/multiclass_results_perclass.csv")

    # Per-class F1 plot
    plot_perclass_f1_multiclass(df_perclass, all_classes)

    # ── Summary tables ────────────────────────────────────
    print("\n" + "═"*60)
    print("  MULTICLASS — OVERALL SUMMARY (sorted by ROC-AUC OvR)")
    print("═"*60)
    print(df_overall.to_string(index=False, float_format="{:.4f}".format))

    print("\n" + "═"*60)
    print("  MULTICLASS — PER-CLASS F1 SUMMARY")
    print("═"*60)
    f1_cols = (["Model", "Strategy"] +
               [f"F1_{MULTICLASS_LABELS.get(c, c)}" for c in all_classes] +
               ["F1_Macro", "F1_Weighted"])
    f1_cols = [c for c in f1_cols if c in df_perclass.columns]
    print(df_perclass[f1_cols].to_string(
        index=False, float_format="{:.4f}".format))

    print("\n" + "═"*60)
    print("  MULTICLASS — PER-CLASS PRECISION SUMMARY")
    print("═"*60)
    prec_cols = (["Model", "Strategy"] +
                 [f"Precision_{MULTICLASS_LABELS.get(c, c)}" for c in all_classes] +
                 ["Precision_Macro", "Precision_Weighted"])
    prec_cols = [c for c in prec_cols if c in df_perclass.columns]
    print(df_perclass[prec_cols].to_string(
        index=False, float_format="{:.4f}".format))

    print("\n" + "═"*60)
    print("  MULTICLASS — PER-CLASS RECALL SUMMARY")
    print("═"*60)
    rec_cols = (["Model", "Strategy"] +
                [f"Recall_{MULTICLASS_LABELS.get(c, c)}" for c in all_classes] +
                ["Recall_Macro", "Recall_Weighted"])
    rec_cols = [c for c in rec_cols if c in df_perclass.columns]
    print(df_perclass[rec_cols].to_string(
        index=False, float_format="{:.4f}".format))

    return df_overall, df_perclass


# ══════════════════════════════════════════════════════════
# MODEL COMPARISON PLOT
# ══════════════════════════════════════════════════════════

def plot_model_comparison(df_bin_pc: pd.DataFrame,
                           df_mc_pc: pd.DataFrame,
                           mc_classes: list) -> None:
    """Side-by-side per-class F1 heatmaps for easy comparison."""
    bin_cls   = [BINARY_LABELS[0], BINARY_LABELS[1]]
    mc_cls    = [MULTICLASS_LABELS.get(c, str(c)) for c in mc_classes]

    bin_f1_cols = [f"F1_{c}" for c in bin_cls
                   if f"F1_{c}" in df_bin_pc.columns]
    mc_f1_cols  = [f"F1_{c}" for c in mc_cls
                   if f"F1_{c}" in df_mc_pc.columns]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(df_bin_pc)*0.5 + 2)))
    fig.suptitle("Per-Class F1 Heatmap — All Models & Strategies",
                 fontsize=13, fontweight="bold")

    # Binary heatmap
    if bin_f1_cols:
        mat_bin = df_bin_pc[bin_f1_cols].values.astype(float)
        sns.heatmap(mat_bin, annot=True, fmt=".3f", cmap="RdYlGn",
                    xticklabels=[c.replace("F1_","") for c in bin_f1_cols],
                    yticklabels=df_bin_pc["Model"].tolist(),
                    vmin=0, vmax=1, linewidths=0.5, ax=axes[0],
                    annot_kws={"size": 8})
        axes[0].set_title("Binary — Per-Class F1")
        axes[0].tick_params(axis="y", labelsize=7)

    # Multiclass heatmap
    if mc_f1_cols:
        mat_mc = df_mc_pc[mc_f1_cols].values.astype(float)
        sns.heatmap(mat_mc, annot=True, fmt=".3f", cmap="RdYlGn",
                    xticklabels=[c.replace("F1_","") for c in mc_f1_cols],
                    yticklabels=df_mc_pc["Model"].tolist(),
                    vmin=0, vmax=1, linewidths=0.5, ax=axes[1],
                    annot_kws={"size": 8})
        axes[1].set_title("Multiclass — Per-Class F1")
        axes[1].tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    plt.savefig("results/figures/model_comparison_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Model comparison heatmap → "
          "results/figures/model_comparison_heatmap.png")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PHASE 7 — EVALUATION  (Test Set)")
    print("=" * 60)

    os.makedirs("results/tables",  exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    df_bin_overall, df_bin_pc = evaluate_binary()
    df_mc_overall,  df_mc_pc  = evaluate_multiclass()

    # ── Determine multiclass classes for comparison plot ──
    (_, _, X_te_sc_mc, X_te_un_mc,
     _, y_te_mc, _) = load_multiclass_data()
    mc_classes = sorted(np.unique(y_te_mc).astype(int))

    plot_model_comparison(df_bin_pc, df_mc_pc, mc_classes)

    # ── Best models ───────────────────────────────────────
    _separator("BEST MODELS")

    # Binary best = highest ROC-AUC
    best_bin = df_bin_overall.iloc[0]
    # Multiclass best = highest ROC-AUC OvR
    best_mc  = df_mc_overall.iloc[0]

    print(f"\n  Best Binary     : {best_bin['Model']}")
    print(f"    Accuracy={best_bin['Accuracy']:.4f}  "
          f"ROC-AUC={best_bin['ROC_AUC']:.4f}")

    print(f"\n  Best Multiclass : {best_mc['Model']}")
    print(f"    Accuracy={best_mc['Accuracy']:.4f}  "
          f"ROC-AUC(OvR)={best_mc['ROC_AUC_OvR']:.4f}")

    # Save best model names for Phase 8
    pd.Series({
        "best_binary"    : best_bin["Model"],
        "best_multiclass": best_mc["Model"],
    }).to_csv("data/processed/best_model_names.csv", header=False)

    print("\n✅ Phase 7 complete.")
    print("   results/tables/binary_results_overall.csv")
    print("   results/tables/binary_results_perclass.csv")
    print("   results/tables/multiclass_results_overall.csv")
    print("   results/tables/multiclass_results_perclass.csv")
    print("   results/figures/  ← confusion matrices, ROC, F1 charts, heatmap")


if __name__ == "__main__":
    main()